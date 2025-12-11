"""
Google Slides Recap Generator

Generates a formatted recap slide in Google Slides with Command of the Message sections:
- Current State
- Future State
- Negative Consequences
- Positive Business Outcomes
- Required Capabilities
"""

import os
import json
from typing import Optional
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from models import RecapSlideData


class GoogleSlidesGenerator:
    """Generates recap slides in Google Slides using service account authentication."""
    
    SCOPES = [
        'https://www.googleapis.com/auth/presentations',
        'https://www.googleapis.com/auth/drive'
    ]
    
    def __init__(self):
        """Initialize the Google Slides generator with service account credentials."""
        self.credentials = None
        self.slides_service = None
        self.drive_service = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate using service account credentials from environment variable."""
        creds_json = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON')
        
        if not creds_json:
            raise ValueError(
                "GOOGLE_SERVICE_ACCOUNT_JSON environment variable is not set. "
                "Please provide service account credentials as a JSON string."
            )
        
        try:
            creds_info = json.loads(creds_json)
            self.credentials = service_account.Credentials.from_service_account_info(
                creds_info,
                scopes=self.SCOPES
            )
            self.slides_service = build('slides', 'v1', credentials=self.credentials)
            self.drive_service = build('drive', 'v3', credentials=self.credentials)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in GOOGLE_SERVICE_ACCOUNT_JSON: {e}")
        except Exception as e:
            raise ValueError(f"Failed to authenticate with Google: {e}")
    
    def create_recap_presentation(self, data: RecapSlideData) -> str:
        """
        Create a new Google Slides presentation with a recap slide.
        
        Args:
            data: RecapSlideData containing the content for the slide
            
        Returns:
            URL of the created presentation
        """
        try:
            # Create a new presentation
            title = f"Call Recap: {data.customer_name}" if data.customer_name else "Call Recap"
            presentation = self.slides_service.presentations().create(
                body={'title': title}
            ).execute()
            
            presentation_id = presentation['presentationId']
            
            # Get the default slide ID (first slide created automatically)
            slides = presentation.get('slides', [])
            if slides:
                first_slide_id = slides[0]['objectId']
                # Delete the default title slide
                self.slides_service.presentations().batchUpdate(
                    presentationId=presentation_id,
                    body={
                        'requests': [{'deleteObject': {'objectId': first_slide_id}}]
                    }
                ).execute()
            
            # Create our custom recap slide
            self._create_recap_slide(presentation_id, data)
            
            # Make the presentation viewable by anyone with the link
            self._set_sharing_permissions(presentation_id)
            
            return f"https://docs.google.com/presentation/d/{presentation_id}/edit"
            
        except HttpError as e:
            raise Exception(f"Google Slides API error: {e}")
    
    def _create_recap_slide(self, presentation_id: str, data: RecapSlideData):
        """Create the formatted recap slide with all sections."""
        
        # Generate unique IDs for elements
        slide_id = 'recap_slide_1'
        
        requests = []
        
        # Create a blank slide
        requests.append({
            'createSlide': {
                'objectId': slide_id,
                'slideLayoutReference': {'predefinedLayout': 'BLANK'}
            }
        })
        
        # Title
        title_text = f"Call Recap: {data.customer_name}" if data.customer_name else "Call Recap"
        if data.call_date:
            title_text += f"\n{data.call_date}"
        
        requests.extend(self._create_text_box(
            slide_id, 'title_box', title_text,
            left=50, top=20, width=620, height=50,
            font_size=24, bold=True, alignment='CENTER'
        ))
        
        # Current State (top left)
        current_state_text = "CURRENT STATE\n" + "\n".join(f"• {item}" for item in data.current_state) if data.current_state else "CURRENT STATE\n• (No data)"
        requests.extend(self._create_text_box(
            slide_id, 'current_state_box', current_state_text,
            left=25, top=80, width=330, height=140,
            font_size=11, header_size=14
        ))
        
        # Future State (top right)
        future_state_text = "FUTURE STATE\n" + "\n".join(f"• {item}" for item in data.future_state) if data.future_state else "FUTURE STATE\n• (No data)"
        requests.extend(self._create_text_box(
            slide_id, 'future_state_box', future_state_text,
            left=365, top=80, width=330, height=140,
            font_size=11, header_size=14
        ))
        
        # Negative Consequences (middle left)
        neg_consequences_text = "NEGATIVE CONSEQUENCES\n" + "\n".join(f"• {item}" for item in data.negative_consequences) if data.negative_consequences else "NEGATIVE CONSEQUENCES\n• (No data)"
        requests.extend(self._create_text_box(
            slide_id, 'neg_consequences_box', neg_consequences_text,
            left=25, top=230, width=330, height=140,
            font_size=11, header_size=14
        ))
        
        # Positive Business Outcomes (middle right)
        pos_outcomes_text = "POSITIVE BUSINESS OUTCOMES\n" + "\n".join(f"• {item}" for item in data.positive_business_outcomes) if data.positive_business_outcomes else "POSITIVE BUSINESS OUTCOMES\n• (No data)"
        requests.extend(self._create_text_box(
            slide_id, 'pos_outcomes_box', pos_outcomes_text,
            left=365, top=230, width=330, height=140,
            font_size=11, header_size=14
        ))
        
        # Required Capabilities (bottom, full width)
        req_caps_text = "REQUIRED CAPABILITIES\n" + "\n".join(f"• {item}" for item in data.required_capabilities) if data.required_capabilities else "REQUIRED CAPABILITIES\n• (No data)"
        requests.extend(self._create_text_box(
            slide_id, 'req_caps_box', req_caps_text,
            left=25, top=380, width=670, height=120,
            font_size=11, header_size=14
        ))
        
        # Execute all requests
        self.slides_service.presentations().batchUpdate(
            presentationId=presentation_id,
            body={'requests': requests}
        ).execute()
    
    def _create_text_box(
        self,
        slide_id: str,
        element_id: str,
        text: str,
        left: int,
        top: int,
        width: int,
        height: int,
        font_size: int = 12,
        header_size: int = 14,
        bold: bool = False,
        alignment: str = 'START'
    ) -> list:
        """Create requests for a text box with styling."""
        
        requests = [
            # Create the shape
            {
                'createShape': {
                    'objectId': element_id,
                    'shapeType': 'TEXT_BOX',
                    'elementProperties': {
                        'pageObjectId': slide_id,
                        'size': {
                            'width': {'magnitude': width, 'unit': 'PT'},
                            'height': {'magnitude': height, 'unit': 'PT'}
                        },
                        'transform': {
                            'scaleX': 1,
                            'scaleY': 1,
                            'translateX': left,
                            'translateY': top,
                            'unit': 'PT'
                        }
                    }
                }
            },
            # Insert text
            {
                'insertText': {
                    'objectId': element_id,
                    'text': text,
                    'insertionIndex': 0
                }
            },
            # Style the text
            {
                'updateTextStyle': {
                    'objectId': element_id,
                    'style': {
                        'fontSize': {'magnitude': font_size, 'unit': 'PT'},
                        'bold': bold
                    },
                    'textRange': {'type': 'ALL'},
                    'fields': 'fontSize,bold'
                }
            },
            # Style the header (first line) if different
            {
                'updateTextStyle': {
                    'objectId': element_id,
                    'style': {
                        'fontSize': {'magnitude': header_size, 'unit': 'PT'},
                        'bold': True,
                        'foregroundColor': {
                            'opaqueColor': {
                                'rgbColor': {'red': 0.2, 'green': 0.2, 'blue': 0.6}
                            }
                        }
                    },
                    'textRange': {
                        'type': 'FIXED_RANGE',
                        'startIndex': 0,
                        'endIndex': text.find('\n') if '\n' in text else len(text)
                    },
                    'fields': 'fontSize,bold,foregroundColor'
                }
            },
            # Set paragraph alignment
            {
                'updateParagraphStyle': {
                    'objectId': element_id,
                    'style': {'alignment': alignment},
                    'textRange': {'type': 'ALL'},
                    'fields': 'alignment'
                }
            }
        ]
        
        return requests
    
    def _set_sharing_permissions(self, presentation_id: str):
        """Set the presentation to be viewable by anyone with the link."""
        try:
            self.drive_service.permissions().create(
                fileId=presentation_id,
                body={
                    'type': 'anyone',
                    'role': 'reader'
                }
            ).execute()
        except HttpError as e:
            # Log but don't fail if sharing permissions can't be set
            print(f"Warning: Could not set sharing permissions: {e}")


def generate_recap_slide(data: RecapSlideData) -> str:
    """
    Convenience function to generate a recap slide.
    
    Args:
        data: RecapSlideData containing the content
        
    Returns:
        URL of the created Google Slides presentation
    """
    generator = GoogleSlidesGenerator()
    return generator.create_recap_presentation(data)
