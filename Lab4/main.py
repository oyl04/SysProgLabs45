import re
import itertools


class Terminal:
    def __init__(self, text):
        self.text = text
        if self.is_empty():  # Replace the text with 'eps' if it represents an empty terminal
            self.text = 'eps'

    # Check if the terminal symbol is an epsilon (empty symbol).
    def is_empty(self):
        return self.text in ('ε', 'epsilon', 'eps')

    def __str__(self) -> str:
        return self.text

    def __eq__(self, o: object) -> bool:
        return self.text == o.text

    def __hash__(self) -> int:
        return hash(self.text)


class NonTerminal:
    def __init__(self, text):
        self.text = text
        # A list to hold production rules for this non-terminal.
        self.rules = []

    def __str__(self) -> str:
        return self.text

    def __eq__(self, o: object) -> bool:
        return o.text == self.text

    def __hash__(self) -> int:
        return hash(self.text)


class Grammar:

    def __init__(self, rules):
        self.regex = None
        self.terminals = []
        self.non_terminals = []
        # Process the given rules to populate terminals and non-terminals.
        self.get_rules(rules)
        self.rules = rules
        # Identify non-terminals that can produce an epsilon (empty string).
        self.epsilon_producers = self.find_epsilon_producing_non_terminals()

    def get_epsilon(self):
        # Find a terminal representing an epsilon.
        eps = next((t for t in self.terminals if t.is_empty()), None)
        if eps is None:
            eps = Terminal('eps')
        return eps

    @staticmethod
    # Generate a regex pattern to identify terminals and non-terminals in text.
    def get_regex(text_set):
        return f"({'|'.join(sorted([re.escape(t.text) for t in text_set], key=lambda x: len(x), reverse=True))})"

    # Process the given rules to populate terminals and non-terminals.
    def get_rules(self, rules):
        # Add non-terminals defined in the rules to the grammar's non_terminals list.
        self.non_terminals += [NonTerminal(n) for n in list(rules.keys())]
        # Create a regular expression pattern to identify non-terminals.
        n_regex = Grammar.get_regex(self.non_terminals)

        terminals = set()
        for n in self.non_terminals:
            for nt_rule in rules[n.text]:
                # Split each production rule and identify terminals.
                g = nt_rule.split()
                # Add identified terminals to the set.
                t = [re.sub(n_regex, ' ', p).split() for p in g]
                terminals |= set(itertools.chain.from_iterable(t))
        # Create Terminal objects for each identified terminal symbol.
        self.terminals = [Terminal(term) for term in terminals]
        # Print the identified terminals and non-terminals.
        print(self.terminals, self.non_terminals)

        # Create a regular expression pattern to identify terminals.
        t_regex = Grammar.get_regex(self.terminals)

        # Combine the terminal and non-terminal patterns into a single pattern.
        nt_regex = f"{n_regex}|{t_regex}"
        self.regex = nt_regex

        # Process each rule for each non-terminal and store in their 'rules' attribute.
        for n in self.non_terminals:
            for nt_rule in rules[n.text]:
                n.rules.append([])
                for m in re.finditer(nt_regex, nt_rule):
                    # Depending on whether the match is a terminal or non-terminal, add it to the rule.
                    if m.group(1):
                        n.rules[-1].append(next((nt for nt in self.non_terminals if nt.text == m.group(1))))
                    elif m.group(2):
                        n.rules[-1].append(next((t for t in self.terminals if t.text == m.group(2))))

    def find_epsilon_producing_non_terminals(self):
        epsilon_producers = set()
        # Check each non-terminal's production rules.
        for nt in self.non_terminals:
            for production in nt.rules:
                # If any production of a non-terminal directly produces an epsilon, add it to the set.
                if any(isinstance(symbol, Terminal) and symbol.is_empty() for symbol in production):
                    epsilon_producers.add(nt)
                    break

        # Continuously check for non-terminals that can indirectly produce an epsilon.
        changed = True
        while changed:
            changed = False
            for nt in self.non_terminals:
                if nt in epsilon_producers:
                    continue
                for production in nt.rules:
                    # If all symbols in a production are epsilon producers or empty terminals, add the non-terminal.
                    if all(symbol in epsilon_producers or (isinstance(symbol, Terminal) and symbol.is_empty()) for
                           symbol in production):
                        epsilon_producers.add(nt)
                        changed = True
                        break

        return epsilon_producers

    # Method to read grammar rules from a file.
    @staticmethod
    def read_grammar_from_file(file_path):
        rules = {}

        with open(file_path, 'r') as file:
            for line in file:
                # Split each line at the '->' symbol to separate the left and right parts of the production rule.
                parts = line.strip().split('->')
                if len(parts) == 2:
                    left, right = parts
                    # Trim whitespace from the left-hand side (non-terminal) of the rule.
                    left = left.strip()
                    # Split the right-hand side of the rule into individual productions using '|' as the delimiter.
                    right_productions = right.strip().split("|")
                    # If the non-terminal is not already in the rules dictionary, add it with its productions.
                    # Otherwise, extend its list of productions with the new ones.
                    if left not in rules:
                        rules[left] = right_productions
                    else:
                        rules[left].extend(right_productions)

        for key in rules.keys():
            rules[key] = ['ε' if prod == 'epsilon' else prod for prod in rules[key]]

        return rules


if __name__ == "__main__":
    print("TEST 1:")
    grammar = Grammar(Grammar.read_grammar_from_file("in.txt"))
