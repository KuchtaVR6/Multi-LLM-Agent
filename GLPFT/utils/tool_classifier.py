import re


# def evaluate_reasoning(reasoning, expected_api, apis):
#     number_of_apis_mentioned = 0
#     for api in apis:
#         if api in reasoning:
#             number_of_apis_mentioned += 1
#
#     if expected_api:
#         correct_api_mentioned = expected_api in reasoning
#         if correct_api_mentioned:
#             if number_of_apis_mentioned == 1:
#                 return 'correct'
#             else:
#                 return 'ambiguous'
#         else:
#             if number_of_apis_mentioned == 0:
#                 return 'no_apis'
#             else:
#                 return 'wrong_apis'
#     else:
#         if number_of_apis_mentioned == 0:
#             return 'partially_correct_no_mention'
#         else:
#             return 'apis_present_but_not_expected'


def remove_negations(text, hard=False):
    for negation in ToolClassifier.negations:
        text = re.sub(negation, '[negated]', text, flags=re.IGNORECASE)
    if hard:
        print('hi!')
        for negation in ToolClassifier.likely_negations:
            print(negation)
            text = re.sub(negation, '[likely negated]', text, flags=re.IGNORECASE)
    return text


class ToolClassifier:
    negations = [
        r' is not the [^ ]+? ',
        r' [^ ]+? returned an error',
        r' function [^ ]+? failed',
        r' function [^ ]+? also failed',
        r'an error executing the [^ ]+? ',
        r' [^ ]+? function is not working',
    ]

    likely_negations = [
        r'attempted to use [^ ]+? ',
        r'attempted to use the [^ ]+? ',
        r'I tried to use the [^ ]+? ',
        r'it seems that the function [^ ]+? ',
        r'I initially called the [^ ]+? ',
        r'Additionally, I will call the function [^ ]+? ',
        r'2. Use the function [^ ]+? ',
        r'I called the API function [^ ]+? ',
        r'I called the [^ ]+? '
    ]

    def __init__(self, tools_available, verbose=False):
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
                tool
            ]
        self.verbose = verbose
        self.encountered = set()

    def tool_string_lookup(self, text):
        tools_matched = set()
        for tool, strings in self.string_matching_tools.items():
            for string in strings:
                if string in text:
                    tools_matched.add(tool)
        return list(tools_matched)

    def feed_plan(self, plan):
        self.encountered = self.tool_string_lookup(remove_negations(plan))

        if len(self.encountered) == 1:
            print('✅')
            return self.encountered[0]
        else:
            if len(self.encountered) > 1:
                hard_removed = self.tool_string_lookup(remove_negations(plan, hard=True))
                if len(hard_removed) == 1:
                    print('✅')
                    return hard_removed[0]
        return None

