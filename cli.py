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
        if not os.path.isdir(document.text):
            raise ValidationError(
                message="Please enter a valid dircetory path",
                cursor_position=len(document.text))


class FileValidator(Validator):
    """
    Class for custom validator to validate files
    """

    def validate(self, document):
        if not os.path.isfile(document.text):
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
    style = style_from_dict({
        Token.QuestionMark: "#45ed18 bold",
        Token.Selected: "#673ab8 bold",
        Token.Instruction: "",  # default
        Token.Answer: "#2177f4 bold",
        Token.Question: "",
    })
    # style = style_from_dict({
    #     Token.Separator: "#6C6C6C",
    #     Token.QuestionMark: "#FF9D00 bold",
    #     # Token.Selected: "",  # default
    #     Token.Selected: "#5F819D",
    #     Token.Pointer: "#FF9D00 bold",
    #     Token.Instruction: "",  # default
    #     Token.Answer: "#5F819D bold",
    #     Token.Question: "",
    # })
    # import pdb; pdb.set_trace()

    questions = [
        {
            "type": "list",
            "name": "image_or_directory",
            "message": "Do you want to modify and image or a directory",
            "choices": ["image", "directory"],
            "filter": lambda val: val.lower(),
        },
        {
            "type": "input",
            "name": "content_path",
            "message": "What is the path to the content image?",
            "when": lambda answers: answers["image_or_directory"] == "image",
            "validate": FileValidator,
        },
        {
            "type": "list",
            "name": "content_directory",
            "message": "Please choose the directory you want to modify, or select other",
            "choices": [file for file in os.listdir(directory) \
                                if os.path.isdir(file)]+["other"],
            "when": lambda answers: answers["image_or_directory"] == "directory",
            # "validate": DirectoryValidator,
        },
        {
            "type": "input",
            "name": "content_directory_other",
            "message": "What is the path to directory of content images?",
            "when": lambda answers: answers["content_directory"] == "other",
            "validate": DirectoryValidator,
        },
        {
            "type": "list",
            "name": "style_path",
            "message": "Please choose the style image you want to modify, or select other",
            "choices": [file for file in os.listdir(f"{directory}/style_images") \
                            if ".png" in file or ".jpg" in file]+["other"],
            "filter": lambda val: os.path.join(directory, "style_images", val) \
                                    if val != "other" else val
            # "validate": DirectoryValidator,
        },
        {
            "type": "input",
            "name": "style_path_other",
            "message": "What is the path to the style image?",
            "when": lambda answers: answers["style_path"] == "other",
            "validate": FileValidator,
        },
        {
            "type": "list",
            "name": "iterations",
            "message": "How finely do you want the style to be transfered?",
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
                        "(Please don\'t select high on cloud 9)",
            "choices": ["High", "Medium", "Low"],
            "filter": lambda val: 1500 if val == "High" else 512 if val == "Medium" else 256,
        },
    ]
    return {"questions": questions, "style": style}
