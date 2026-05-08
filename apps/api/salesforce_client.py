"""
Salesforce REST access for server-side integrations.

Supports (in order):
1. Username + password + security token via SOAP Partner login (no Connected App).
2. Optional short-lived access token from SF CLI for local dev (see ``from_sf_cli``).
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import quote, urljoin

import requests

logger = logging.getLogger("sa-call-analyzer")

_NS_SOAP = "http://schemas.xmlsoap.org/soap/envelope/"
_NS_SF = "urn:partner.soap.sforce.com"


def _local(tag: str) -> str:
    return f"{{{_NS_SF}}}{tag}"


def _soap_envelope(username: str, password_with_token: str) -> str:
    """Build SOAP login body; escape XML special chars in credentials."""
    from xml.sax.saxutils import escape

    u = escape(username)
    p = escape(password_with_token)
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        "<env:Envelope xmlns:env=\"http://schemas.xmlsoap.org/soap/envelope/\" "
        'xmlns:urn="urn:partner.soap.sforce.com">'
        "<env:Body>"
        "<urn:login>"
        f"<urn:username>{u}</urn:username>"
        f"<urn:password>{p}</urn:password>"
        "</urn:login>"
        "</env:Body>"
        "</env:Envelope>"
    )


def _parse_soap_login(xml_text: str) -> tuple[str, str]:
    root = ET.fromstring(xml_text)
    # Fault?
    fault = root.find(f".//{{{_NS_SOAP}}}Fault")
    if fault is not None:
        fault_string = fault.findtext("faultstring") or fault.text or "SOAP Fault"
        raise RuntimeError(fault_string.strip())

    body = root.find(f".//{{{_NS_SOAP}}}Body")
    if body is None:
        raise RuntimeError("Invalid SOAP response: missing Body")

    result = body.find(_local("loginResponse"))
    if result is None:
        raise RuntimeError("Invalid SOAP response: missing loginResponse")

    lr = result.find(_local("result"))
    if lr is None:
        raise RuntimeError("Invalid SOAP response: missing login result")

    session_id = lr.findtext(_local("sessionId"))
    server_url = lr.findtext(_local("serverUrl"))
    if not session_id or not server_url:
        raise RuntimeError("Invalid SOAP response: missing sessionId or serverUrl")

    # serverUrl is like https://na123.salesforce.com/services/Soap/u/66.0
    m = re.match(r"^(https://[^/]+)", server_url)
    instance_url = m.group(1) if m else server_url.rsplit("/services", 1)[0]
    return session_id.strip(), instance_url.strip()


class SalesforceClient:
    """Minimal REST + SOQL client."""

    def __init__(self, session_id: str, instance_url: str, api_version: str = "59.0"):
        self.session_id = session_id
        self.instance_url = instance_url.rstrip("/")
        v = (api_version or "59.0").strip()
        self.api_version = v if v.startswith("v") else f"v{v}"

    @classmethod
    def from_password_flow(cls) -> Optional["SalesforceClient"]:
        username = (os.getenv("SALESFORCE_USERNAME") or "").strip()
        password = (os.getenv("SALESFORCE_PASSWORD") or "").strip()
        token = (os.getenv("SALESFORCE_SECURITY_TOKEN") or "").strip()
        if not username or not password:
            return None
        login_url = (os.getenv("SALESFORCE_LOGIN_URL") or "https://login.salesforce.com").rstrip("/")
        api_version = (os.getenv("SALESFORCE_API_VERSION") or "59.0").strip()
        soap_u = api_version[1:] if api_version.startswith("v") else api_version

        pwd = f"{password}{token}" if token else password
        url = f"{login_url}/services/Soap/u/{soap_u}"
        body = _soap_envelope(username, pwd)
        resp = requests.post(
            url,
            data=body.encode("utf-8"),
            headers={
                "Content-Type": "text/xml; charset=UTF-8",
                "SOAPAction": '""',
            },
            timeout=30,
        )
        if not resp.ok:
            raise RuntimeError(f"Salesforce SOAP login HTTP {resp.status_code}: {resp.text[:500]}")
        sid, inst = _parse_soap_login(resp.text)
        return cls(sid, inst, api_version=api_version if api_version.startswith("v") else f"v{api_version}")

    @classmethod
    def from_sf_cli(cls, target_org: Optional[str] = None) -> Optional["SalesforceClient"]:
        """Use ``sf org display`` access token (local machine only)."""
        if os.getenv("VERCEL") == "1":
            return None
        if os.getenv("SALESFORCE_USE_SF_CLI", "").strip().lower() in ("0", "false", "no"):
            return None
        if not shutil.which("sf"):
            return None
        alias = (target_org or os.getenv("SALESFORCE_SF_CLI_TARGET_ORG") or "arize-sfdc").strip()
        try:
            proc = subprocess.run(
                ["sf", "org", "display", "user", "--target-org", alias, "--json"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except FileNotFoundError:
            return None
        except subprocess.SubprocessError as e:
            logger.warning("sf org display failed: %s", e)
            return None
        if proc.returncode != 0:
            logger.warning("sf org display exit %s: %s", proc.returncode, (proc.stderr or "")[:300])
            return None
        try:
            data = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError:
            return None
        result = data.get("result") or {}
        token = (result.get("accessToken") or result.get("access_token") or "").strip()
        inst = (result.get("instanceUrl") or result.get("instance_url") or "").strip().rstrip("/")
        api_version = (os.getenv("SALESFORCE_API_VERSION") or "59.0").strip()
        if not token or not inst:
            return None
        v = api_version if api_version.startswith("v") else f"v{api_version}"
        return cls(token, inst, api_version=v)

    @classmethod
    def from_env(cls) -> Optional["SalesforceClient"]:
        try:
            c = cls.from_password_flow()
            if c:
                return c
        except Exception as e:
            logger.warning("Salesforce password/token login failed: %s", e)
        return cls.from_sf_cli()

    def query(self, soql: str) -> list[dict[str, Any]]:
        """Run SOQL and return all records (follows nextRecordsUrl)."""
        base = f"{self.instance_url}/services/data/{self.api_version}/query"
        out: list[dict[str, Any]] = []
        url: Optional[str] = f"{base}?q={quote(soql)}"
        headers = {"Authorization": f"Bearer {self.session_id}", "Accept": "application/json"}

        while url:
            resp = requests.get(url, headers=headers, timeout=60)
            if not resp.ok:
                raise RuntimeError(f"Salesforce query HTTP {resp.status_code}: {resp.text[:800]}")
            payload = resp.json()
            recs = payload.get("records") or []
            for r in recs:
                if isinstance(r, dict):
                    r.pop("attributes", None)
                    out.append(r)
            nxt = payload.get("nextRecordsUrl")
            if nxt:
                if isinstance(nxt, str) and (nxt.startswith("http://") or nxt.startswith("https://")):
                    url = nxt
                else:
                    url = urljoin(self.instance_url + "/", str(nxt).lstrip("/"))
            else:
                url = None

        return out

    def pipeline_user_options(self) -> list[dict[str, Any]]:
        """Distinct users who are Assigned SA (account or opp), Assigned Solutions, or Opp owner (live SOQL)."""
        queries = [
            "SELECT Id, Name FROM User WHERE Id IN "
            "(SELECT Assigned_SA__c FROM Account WHERE Assigned_SA__c != null AND IsDeleted = false)",
            "SELECT Id, Name FROM User WHERE Id IN "
            "(SELECT Assigned_SA__c FROM Opportunity WHERE Assigned_SA__c != null AND IsDeleted = false)",
            "SELECT Id, Name FROM User WHERE Id IN "
            "(SELECT Assigned_Solutions__c FROM Opportunity WHERE Assigned_Solutions__c != null AND IsDeleted = false)",
            "SELECT Id, Name FROM User WHERE Id IN "
            "(SELECT OwnerId FROM Opportunity WHERE IsClosed = false AND IsDeleted = false)",
        ]
        seen: dict[str, str] = {}
        for soql in queries:
            try:
                for r in self.query(soql):
                    uid = (r.get("Id") or "").strip()
                    if uid and uid not in seen:
                        seen[uid] = (r.get("Name") or "").strip() or uid
            except Exception:
                pass
        out = [{"id": uid, "name": nm} for uid, nm in seen.items()]
        out.sort(key=lambda x: x["name"].lower())
        return out

    def opportunities_for_pipeline_user(self, user_id: str) -> list[dict[str, Any]]:
        """Open opps where the user is Assigned SA (account or opp level), Assigned Solutions, or Owner."""
        uid = user_id.replace("'", "\\'")
        soql = (
            "SELECT Id, Name, StageName, Amount, CloseDate, NextStep, AccountId, "
            "Account.Name, Owner.Name, LastStageChangeDate "
            "FROM Opportunity "
            f"WHERE IsClosed = false AND ("
            f"Account.Assigned_SA__c = '{uid}' OR "
            f"Assigned_SA__c = '{uid}' OR "
            f"Assigned_Solutions__c = '{uid}' OR "
            f"OwnerId = '{uid}') "
            "ORDER BY CloseDate ASC"
        )
        rows = self.query(soql)
        now = datetime.now(timezone.utc)
        normalized: list[dict[str, Any]] = []
        for r in rows:
            acct = r.get("Account") if isinstance(r.get("Account"), dict) else {}
            if isinstance(acct, dict):
                acct.pop("attributes", None)
            owner = r.get("Owner") if isinstance(r.get("Owner"), dict) else {}
            if isinstance(owner, dict):
                owner.pop("attributes", None)
            days_in_stage: Optional[int] = None
            lscd = r.get("LastStageChangeDate")
            if isinstance(lscd, str) and lscd.strip():
                # Salesforce returns ISO 8601, e.g. "2026-04-12T18:43:21.000+0000"
                try:
                    s = lscd.replace("Z", "+0000")
                    dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f%z")
                    days_in_stage = max((now - dt).days, 0)
                except ValueError:
                    try:
                        dt = datetime.fromisoformat(lscd.replace("Z", "+00:00"))
                        days_in_stage = max((now - dt).days, 0)
                    except ValueError:
                        days_in_stage = None
            normalized.append(
                {
                    "id": r.get("Id"),
                    "name": r.get("Name"),
                    "stage_name": r.get("StageName"),
                    "amount": r.get("Amount"),
                    "close_date": r.get("CloseDate"),
                    "next_step": r.get("NextStep"),
                    "account_id": r.get("AccountId"),
                    "account_name": (acct or {}).get("Name") if isinstance(acct, dict) else None,
                    "owner_name": (owner or {}).get("Name") if isinstance(owner, dict) else None,
                    "days_in_stage": days_in_stage,
                }
            )
        return normalized


_sf_singleton: Optional[SalesforceClient] = None
_sf_singleton_attempted = False
_sf_last_error: Optional[str] = None


def get_cached_salesforce_client() -> Optional[SalesforceClient]:
    """Lazy singleton; retries once if previous attempt failed (creds may have been updated)."""
    global _sf_singleton, _sf_singleton_attempted, _sf_last_error
    if _sf_singleton:
        return _sf_singleton
    if _sf_singleton_attempted:
        return None
    _sf_singleton_attempted = True
    try:
        _sf_singleton = SalesforceClient.from_env()
        _sf_last_error = None
    except Exception as e:
        logger.warning("Salesforce client init failed: %s", e)
        _sf_singleton = None
        _sf_last_error = str(e)
    return _sf_singleton


def get_sf_last_error() -> Optional[str]:
    """Return the last Salesforce login error message (for user-facing diagnostics)."""
    return _sf_last_error


def reset_sf_singleton() -> None:
    """Allow retry of SF login (e.g. after credentials are updated)."""
    global _sf_singleton, _sf_singleton_attempted, _sf_last_error
    _sf_singleton = None
    _sf_singleton_attempted = False
    _sf_last_error = None
