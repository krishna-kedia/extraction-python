"""
extraction_class_type.py

This module defines the data structures used for document extraction
in a structured and type-safe manner using Pydantic models.

Each class represents a specific component of the document, such as
headers and other specific extraction details. You must define the pydantic classes to use
the solution for extraction.
"""

from pydantic import BaseModel, Field
from typing import List


class AIAgentClass(BaseModel):
    information: str = Field(..., description="Given conversation between AI Bot and user in text format from an extraction solution, your job is to convert it to Valid JSON structure provided below. The conversation will be in Hinglish langugage (Hindi written in english script) and you need to extract the information in the same language. The conversation will be between a user and an AI Bot, where the user is providing information about their loan application and the AI Bot is extracting that information.")
    instruction: str = Field(..., description="Given the conversation between AI Bot and user in text format from an extraction solution, your job is to convert it to Valid JSON structure provided below. The field type is also mentioned in with the name of the field in the json. Whatever the language is in the conversation, make sure the information is put in english.")
    condition: str = Field(..., description="Make Sure to extract the relevant answers. Example: If the user says `Mera naam Rajesh Kumar hai`, then extract the name as `Rajesh Kumar`. If the user does not provide any information for a field, then set that field to `NA` (Even if the field type is other than string). This applies to all the fields.")

class ExtractionClass(BaseModel):
    fullName: str = Field(..., description="Full name of the user")
    address: str = Field(..., description="Detailed address of the user")
    loanType: str = Field(..., description="Type of loan requested by the user")
    loanAmount: str = Field(..., description="Amount of loan requested by the user")
    emiComfort: str = Field(..., description="EMI amount the user is comfortable with")
    monthlyObligations: str = Field(..., description="User's current monthly obligations")
    incomeSource: str = Field(..., description="Source of income for the user")
    salaryAmount: str = Field(..., description="Monthly salary of the user")
    designationEmployer: str = Field(..., description="User's designation and employer")
    otherIncomeSources: str = Field(..., description="Other income sources of the user")
    businessRevenue: str = Field(..., description="Monthly revenue from business if applicable")
    businessProfit: str = Field(..., description="Monthly profit from business if applicable")
    businessSalary: str = Field(..., description="Monthly salary from business if applicable")
    atPropertyLocation: str = Field(..., description="Whether the user is at the property location")
    propertyAddress: str = Field(..., description="Address of the property being mortgaged")
    propertyType: str = Field(..., description="Type of the property being mortgaged")
    propertyStructure: str = Field(..., description="Structure of the property being mortgaged")
    propertyUsage: str = Field(..., description="Usage of the property (self-occupied, rented, etc.)")
    landArea: str = Field(..., description="Land area of the property being mortgaged")
    marketValue: str = Field(..., description="Market value of the property being mortgaged")
    existingLoan: str = Field(..., description="Whether there is an existing loan on the property")
    existingLoanEmi: str = Field(..., description="EMI of the existing loan on the property if applicable")


class OutputExampleClass(BaseModel):
    """Example Output"""


__all__ = [
    "ExtractionClass",
    "AIAgentClass",
    "OutputExampleClass"
]