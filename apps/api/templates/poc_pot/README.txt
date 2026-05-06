PoC / PoT Word templates (required for /health poc_pot_workflow.word_templates_present)

Place these three files in BOTH directories checked by the API (see apps/api/main.py /health):

  apps/api/templates/poc_pot/
  apps/api/api/templates/poc_pot/

Files:
  poc_saas.docx
  poc_vpc.docx
  pot.docx

After adding or updating templates, run from the repository root:

  uv run python scripts/patch_arize_docx_links.py

(or use the same interpreter you use for apps/api). That script also syncs copies under apps/api when masters live in templates/poc_pot/ here.

These .docx files are not in git until you commit them. .gitignore does not exclude *.docx — if templates are missing in CI or production, add the files and commit, or mount them at deploy time.
