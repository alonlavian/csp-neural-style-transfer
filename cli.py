import os
from PyInquirer import style_from_dict, Token, prompt
from PyInquirer import Validator, ValidationError


class NumberValidator(Validator):
    def validate(self, document):
        try:
            int(document.text)
        except ValueError:
            raise ValidationError(
                message='Please enter a number',
                cursor_position=len(document.text))  # Move cursor to end


class DirectoryValidator(Validator):
    def validate(self, document):
        if not os.path.isdir(document.text):
            raise ValidationError(
                message='Please enter a valid dircetory path',
                cursor_position=len(document.text))


class FileValidator(Validator):
    def validate(self, document):
        if not os.path.isfile(document.text):
            raise ValidationError(
                message='Please enter a valid file path',
                cursor_position=len(document.text))


def return_cli():
    style = style_from_dict({
        Token.QuestionMark: '#E91E63 bold',
        Token.Selected: '#673AB7 bold',
        Token.Instruction: '',  # default
        Token.Answer: '#2196f3 bold',
        Token.Question: '',
    })
    questions = [
        {
            'type': 'list',
            'name': 'image_or_directory',
            'message': 'Do you want to modify and image or a directory',
            'choices': ["image", "directory"],
            'filter': lambda val: val.lower(),
        },
        {
            'type': 'input',
            'name': 'content_path',
            'message': 'What is the path to the content image?',
            'when': lambda answers: answers['image_or_directory'] == 'image',
            "validate": FileValidator,
        },
        {
            'type': 'input',
            'name': 'content_directory',
            'message': 'What is the path to directory of content images?',
            'when': lambda answers: answers['image_or_directory'] == 'directory',
            "validate": DirectoryValidator,
        },
        {
            'type': 'input',
            'name': 'style_path',
            'message': 'What is the path to the style image?',
            "validate": FileValidator,
        },
        {
            'type': 'input',
            'name': 'iterations',
            'message': 'How many iterations do you want to run?',
            "default": "1000",
            'validate': NumberValidator,
            'filter': lambda val: int(val),
        },
        {
            'type': 'input',
            'name': 'border_size',
            'message': 'How many pixels wide do you want the border?',
            "default": "75",
            'validate': NumberValidator,
            'filter': lambda val: int(val),
        },
        {
            'type': 'input',
            'name': 'max_resolution',
            'message': 'What is the resolution you want to convert the images to?',
            "default": "512",
            'validate': NumberValidator,
            'filter': lambda val: int(val) if int(val) < 1500 else 1500,
        },
    ]
    return {"questions": questions, "style": style}
