"""
Module to package cli functionality of code
"""

import os
import os.path
from PyInquirer import style_from_dict, Token
from PyInquirer import Validator, ValidationError


class NumberValidator(Validator):
    """
    Class for custom validator to validate numbers
    """

    def validate(self, document):
        try:
            int(document.text)
        except ValueError:
            raise ValidationError(
                message="Please enter a number",
                cursor_position=len(document.text))  # Move cursor to end


class DirectoryValidator(Validator):
    """
    Class for custom validator to validate directories
    """

    def validate(self, document):
        if not os.path.isdir(document.text) or document.text == "Other":
            raise ValidationError(
                message="Please enter a valid directory path",
                cursor_position=len(document.text))


class FileValidator(Validator):
    """
    Class for custom validator to validate files
    """

    def validate(self, document):
        if not os.path.isfile(document.text) or document.text == "Other":
            raise ValidationError(
                message="Please enter a valid file path",
                cursor_position=len(document.text))


def return_cli():
    """
    Wrapper function that returns questions and style dict for cli
    Returns:
        A Dict containing questions and style
    """
    directory = os.getcwd()
    # Setting up the style for the CLI
    style = style_from_dict({
        Token.QuestionMark: "#45ed18 bold",
        Token.Selected: "#673ab8 bold",
        Token.Instruction: "",  # default
        Token.Answer: "#2177f4 bold",
        Token.Question: "",
    })
    # Building question config script
    questions = [
        {
            "type": "list",
            "name": "image_or_directory",
            "message": "Do you want to modify and image or a directory?",
            "choices": ["image", "directory"],
            "filter": lambda val: val.lower(),
        },
        {
            "type": "list",
            "name": "content_path",
            "message": "Please choose the style image you want to modify, or select other",
            "choices": [file for file in os.listdir(f"{directory}/content_images") \
                            if ".png" in file or ".jpg" in file]+["Other"],
            "filter": lambda val: os.path.join(directory, "content_images", val) \
                                    if val != "Other" else val,
            "when": lambda answers: answers["image_or_directory"] == "image"
            # "validate": DirectoryValidator,
        },
        {
            "type": "input",
            "name": "content_path",
            "message": "What is the path to the style image?",
            "when": lambda answers: answers.get("content_path", "N/A") == "Other",
            "validate": FileValidator,
        },
        {
            "type": "list",
            "name": "content_directory",
            "message": "Please choose the directory you want to modify, or select other",
            "choices": [file for file in os.listdir(directory) \
                                if os.path.isdir(file)]+["Other"],
            "when": lambda answers: answers["image_or_directory"] == "directory",
            # "validate": DirectoryValidator,
        },
        {
            "type": "input",
            "name": "content_directory",
            "message": "What is the path to directory of content images?",
            "when": lambda answers: answers.get("content_directory", "N/A") == ["Other"],
            "validate": DirectoryValidator,
        },
        {
            "type": "list",
            "name": "style_path",
            "message": "Please choose the style image you want to modify, or select other",
            "choices": [file for file in os.listdir(f"{directory}/style_images") \
                            if ".png" in file or ".jpg" in file]+["Other"],
            "filter": lambda val: os.path.join(directory, "style_images", val) \
                                    if val != "Other" else val
            # "validate": DirectoryValidator,
        },
        {
            "type": "input",
            "name": "style_path",
            "message": "What is the path to the style image?",
            "when": lambda answers: answers["style_path"] == "Other",
            "validate": FileValidator,
        },
        {
            "type": "list",
            "name": "iterations",
            "message": "How finely do you want the style to be transferred?",
            "choices": ["High", "Medium", "Low"],
            "filter": lambda val: 1000 if val == "High" else \
                                  500 if val == "Medium" else 100,
        },
        {
            "type": "input",
            "name": "border_size",
            "message": "How many pixels wide do you want the border?",
            "default": "75",
            "validate": NumberValidator,
            "filter": int,
        },
        {
            "type": "list",
            "name": "max_resolution",
            "message": "What is the resolution you want to convert the images to? " +
                        "(Please don\'t select high on Cloud 9)",
            "choices": ["High", "Medium", "Low"],
            "filter": lambda val: 1500 if val == "High" else 512 if val == "Medium" else 256,
        },
    ]
    return {"questions": questions, "style": style}
