import re
import json


def remove_negations(text, hard=False):
    for negation in ToolClassifier.negations:
        text = re.sub(negation, '[negated]', text, flags=re.IGNORECASE)
    if hard:
        for negation in ToolClassifier.likely_negations:
            text = re.sub(negation, '[likely negated]', text, flags=re.IGNORECASE)
    return text


with open('classification_fail_log.json', 'w') as file:
    file.write('[')


class ToolClassifier:
    negations = [
        r' is not the [^ ]+? ',
        r' the [^ ]+? returned an error',
        r' the [^ ]+? function also failed',
        r' function [^ ]+? also failed',  # has to be after the above case
        r' the [^ ]+? function still failed',
        r' the [^ ]+? function failed',
        r' function [^ ]+? failed',  # has to be after the above cases
        r' an error executing the [^ ]+? ',
        r' the [^ ]+? function is not working',
        r' an issue with the functionality of the [^ ]+? function',
        r'Instead of calling the function [^ ]+?, ',
        r' function [^ ]+? is not valid ',
        r' the [^ ]+? function, I\'ll now call the ',
        r' the [^ ]+? function, I decided to call the',
        r'Based on the response from the [^ ]+? function',
        r'the [^ ]+? function. The response returned an empty object.',
        r'the [^ ]+? function resulted in an error',
        r'function [^ ]+? is currently unavailable',
        r'the [^ ]+? and [^ ]+? functions are currently unavailable',
        r' instead of [^ ]+? ',
        r' [^ ]+? failed',
        r' [^ ]+? was unsuccessful',
        r' while trying to call the [^ ]+? function'
    ]

    likely_negations = [
        r' attempted to use [^ ]+? function ',
        r' attempted to use the [^ ]+? function ',
        r' tried to use the [^ ]+? ',
        r' it seems that the function [^ ]+? ',
        r' it seems that the [^ ]+? function ',
        r' initially called the [^ ]+? ',
        r'Additionally, I will call the function [^ ]+? ',
        r'Additionally, I can use the function [^ ]+? ',
        r'Furthermore, by calling the function [^ ]+? ',
        r'[23456789]. Use the function [^ ]+? ',
        r'[23456789]. Call the [^ ]+? function ',
        r' called the API function [^ ]+? ',
        r' called the [^ ]+? ',
        r' called the [^ ]+? function',
        r' has been called with the function [^ ]+? ',
        r' and then use the function [^ ]+? to ',
        r' previous API call to [^ ]+? ',
        r'the [^ ]+? function resulted',
        r' retrieved using the [^ ]+? function',
    ]

    terminal_symbols = ['`', "'", '"', ' ']

    def __init__(self, tools_available):
        self.tools_available = tools_available
        self.string_matching_tools = {}
        for tool in self.tools_available:
            self.string_matching_tools[tool] = [
                f'`{tool}`',
                f'"{tool}"',
                f"'{tool}'",
                f'`{tool}(',
                f'"{tool}(',
                f"'{tool}(",
                f" {tool} "
            ]
        self.encountered = set()

    def advanced_tool_string_lookup(self, text):
        tools_matched = set()
        for tool, strings in self.string_matching_tools.items():
            for string in strings:
                if string in text:
                    tools_matched.add(tool)
        return list(tools_matched)

    def simple_tool_string_lookup(self, text):  # baseline and the pre 2406 approach
        tools_matched = set()
        for tool in self.string_matching_tools.keys():
            if tool in text:
                tools_matched.add(tool)
        return list(tools_matched)

    def feed_plan(self, plan):
        self.encountered = self.simple_tool_string_lookup(remove_negations(plan))

        if len(self.encountered) == 1:
            return self.encountered[0]
        else:
            if len(self.encountered) > 1:
                hard_removed = self.simple_tool_string_lookup(remove_negations(plan, hard=True))
                if len(hard_removed) == 1:
                    return hard_removed[0]
                elif len(hard_removed) > 0:
                    with open('classification_fail_log.json', 'a') as file:
                        json.dump({
                            'hard_removed': hard_removed,
                            'original': plan,
                            'edited': remove_negations(plan, hard=True)
                        }, file, indent=4)
                        file.write(",\n")
        return None
