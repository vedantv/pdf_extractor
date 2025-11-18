import base64
import io
import json
import os
from typing import Optional, List

import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field
from mistralai import Mistral
from mistralai.extra import response_format_from_pydantic_model
from dotenv import load_dotenv
load_dotenv()


# -------------------- Pydantic Schema -------------------- #

class Claim(BaseModel):
    """
    Normalized schema for all 3 payer types (VCM, Anthem, Centersplan).
    """

    plan_name: str = Field(
        ...,
        description=(
            "Name of the health plan / payer appearing on the document. "
            "Examples: 'VillageCare Max', 'VCM', 'Anthem', 'Centersplan'."
        ),
    )

    patient_name: str = Field(
        ...,
        description=(
            "Full name of the member/patient as written on the document. "
            "Typically near labels like 'Name' or 'Member Name'."
        ),
    )

    member_id: Optional[str] = Field(
        None,
        description=(
            "Member identifier (e.g. 'Member ID', 'Member #'). For Anthem, "
            "map 'Member #' here. For VCM and Centersplan, map 'Member ID'."
        ),
    )

    medicaid_number: Optional[str] = Field(
        None,
        description=(
            "Medicaid number for the member, often labeled 'Medicaid #' or "
            " 'Medicaid No.'."
        ),
    )

    date_of_birth: Optional[str] = Field(
        None,
        description=(
            "Date of birth of the member, typically labeled 'DOB' or "
            "'Date of Birth'. Keep the format exactly as shown in the PDF."
        ),
    )

    authorization_number: Optional[str] = Field(
        None,
        description=(
            "Authorization number for the services. Label variants: "
            "'Auth #', 'Authorization #', 'AUthorization #'."
        ),
    )

    primary_diagnosis: Optional[str] = Field(
        None,
        description=(
            "Primary diagnosis code or description. For VCM this may be under "
            "'DX'; for Anthem 'Primary DX'; for Centersplan 'Primary Diagnosis'."
        ),
    )

    procedure_code: Optional[str] = Field(
        None,
        description=(
            "Procedure / service code. For VCM map 'CPT Code'; for Anthem "
            "map 'Procedure Code'; for Centersplan map 'HCPCS'."
        ),
    )

    modifier: Optional[str] = Field(
        None,
        description=(
            "Any modifier associated with the procedure code, typically labeled "
            "'Modifier'. If multiple, join in one string like '59, GP'."
        ),
    )


# -------------------- Helper Functions -------------------- #

def encode_pdf_from_bytes(file_bytes: bytes) -> str:
    """
    Encode PDF bytes as a base64 data URL string, as required by Mistral OCR.
    """
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:application/pdf;base64,{b64}"


def create_client(api_key: str) -> Mistral:
    return Mistral(api_key=api_key)


def extract_claim_from_file(client: Mistral, file_bytes: bytes, filename: str) -> dict:
    """
    Call Mistral OCR+annotations on a single uploaded PDF and
    return a dict that matches the Claim schema.
    """
    document_url = encode_pdf_from_bytes(file_bytes)

    doc_format = response_format_from_pydantic_model(Claim)

    response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": document_url,
        },
        document_annotation_format=doc_format,
        include_image_base64=False,
    )

    annotation = response.document_annotation

    if isinstance(annotation, str):
        try:
            data = json.loads(annotation)
        except json.JSONDecodeError:
            data = {"raw_document_annotation": annotation}
    elif isinstance(annotation, dict):
        data = annotation
    else:
        try:
            data = annotation.model_dump()
        except Exception:
            data = {"raw_document_annotation": str(annotation)}

    data["source_file"] = filename
    return data


def records_to_dataframe(records: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)

    desired_cols = [
        "source_file",
        "plan_name",
        "patient_name",
        "member_id",
        "medicaid_number",
        "date_of_birth",
        "authorization_number",
        "primary_diagnosis",
        "procedure_code",
        "modifier",
    ]

    cols_present = [c for c in desired_cols if c in df.columns]
    remaining = [c for c in df.columns if c not in cols_present]
    df = df[cols_present + remaining]
    return df


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    buffer.seek(0)
    return buffer.read()


# -------------------- Streamlit App -------------------- #

def main():
    st.set_page_config(page_title="Claims PDF Extractor", layout="wide")

    st.title("📄 Claims PDF Extractor (Mistral + Streamlit)")
    st.write(
        "Upload VCM / Anthem / Centersplan PDFs and extract normalized fields "
        "into a single table, then download as Excel."
    )

    # ----- API Key (from environment variable only) ----- #
    api_key = os.getenv("MISTRAL_API_KEY", "")

    # ----- Main area: file upload ----- #

    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("🔍 Extract data", type="primary"):
        if not api_key:
            st.error("MISTRAL_API_KEY environment variable is not set. Please configure it in your environment.")
            return

        if not uploaded_files:
            st.error("Please upload at least one PDF file.")
            return

        # Create client
        try:
            client = create_client(api_key)
        except Exception as e:
            st.error(f"Error creating Mistral client: {e}")
            return

        records = []
        progress = st.progress(0)
        status_placeholder = st.empty()

        for i, file in enumerate(uploaded_files, start=1):
            status_placeholder.write(f"Processing `{file.name}` ({i}/{len(uploaded_files)})...")
            try:
                file_bytes = file.read()
                record = extract_claim_from_file(client, file_bytes, file.name)
                records.append(record)
            except Exception:
                st.error(f"Error processing `{file.name}`. Please try uploading this PDF again.")
            progress.progress(i / len(uploaded_files))

        status_placeholder.empty()

        if not records:
            st.warning("No records extracted. Check the logs or try different PDFs.")
            return

        df = records_to_dataframe(records)

        st.success("Extraction complete!")
        st.subheader("Extracted Data")
        st.dataframe(df, use_container_width=True)

        # Download as Excel
        excel_bytes = df_to_excel_bytes(df)
        st.download_button(
            label="⬇️ Download as Excel",
            data=excel_bytes,
            file_name="claims_extracted.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


if __name__ == "__main__":
    main()
