from collections import deque
import re
import itertools


class Terminal:
    def __init__(self, text):
        self.text = text
        if self.is_empty():  # Replace the text with 'eps' if it represents an empty terminal
            self.text = 'ε'

    # Check if the terminal symbol is an epsilon (empty symbol).
    def is_empty(self):
        return self.text in ('ε', 'epsilon', 'eps')

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
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

    def __repr__(self) -> str:
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

    def get_tnt_string(self, text):
        result = []
        for m in re.finditer(self.regex, text):
            if m.group(1):
                # Append the corresponding non-terminal object to the result
                result.append(next((nt for nt in self.non_terminals if nt.text == m.group(1))))
            elif m.group(2):
                # Append the corresponding terminal object to the result
                result.append(next((t for t in self.terminals if t.text == m.group(2))))
        return result


class FirstFollow:
    def __init__(self, grammar) -> None:
        self.grammar = grammar

    def compute_first_k(self, k):
        first = {}
        prev_first = {}
        # Initialize the first set for each non-terminal in the grammar to be empty.
        for n in self.grammar.non_terminals:
            first[n] = set()
        # Loop until the first set no longer changes between iterations.
        while first != prev_first:
            prev_first = first.copy()
            # Create a value copy of each set in the dictionary to avoid modifying the original sets during iteration.
            for key, value in prev_first.items():
                prev_first[key] = value.copy()
            for n in self.grammar.non_terminals:
                for rule in n.rules:
                    # Calculate possible strings of length k or less from the rule using the current first set.
                    possible_strings = self.get_possible_strings(rule, k, first)
                    # Update the first set of the non-terminal by adding the new possible strings.
                    first[n] |= set(possible_strings)

        return first

    def compute_follow_k(self, k, first_k):
        follow = {}
        prev_follow = {}
        # Initialize the follow set for each non-terminal in the grammar.
        for n in self.grammar.non_terminals:
            follow[n] = set()
        # For the start symbol, add epsilon to its follow set.
        follow[self.grammar.non_terminals[0]].add((self.grammar.get_epsilon(),))

        # Initialize a set to track which non-terminals have been seen.
        seen_nonterminals = set()
        # Add the start symbol to the seen non-terminals set.
        seen_nonterminals.add(self.grammar.non_terminals[0])

        # Loop until the follow set no longer changes between iterations.
        while follow != prev_follow:
            prev_follow = follow.copy()
            for key, value in prev_follow.items():
                # Create a value copy of each set in the dictionary to
                # avoid modifying the original sets during iteration.
                prev_follow[key] = value.copy()

            # Store newly seen non-terminals during this iteration.
            new_seen_non_terminals = []
            for nt in seen_nonterminals:
                for rule in nt.rules:
                    for i, c in enumerate(rule):
                        # Only process non-terminal symbols.
                        if isinstance(c, NonTerminal):
                            # Add the non-terminal to the buffer of newly seen non-terminals.
                            new_seen_non_terminals.append(c)
                            # Get the symbols following the current non-terminal in the rule.
                            after = rule[i + 1:]
                            # Calculate the first set of the string following the current non-terminal.
                            first_of_after = self.concatenate_k(k, [ps[:k] for ps in
                                                                    self.get_possible_strings(after, k, first_k)],
                                                                follow[nt])
                            # Update the follow set of the current non-terminal.
                            for s in first_of_after:
                                if len(s) == 0:
                                    # If the string is empty, add epsilon.
                                    follow[c].add((self.grammar.get_epsilon(),))
                                else:
                                    follow[c].add(s)

            seen_nonterminals |= set(new_seen_non_terminals)

        return follow

    def concatenate_k(self, k, first_set, second_set):
        result = set()
        for s1 in first_set:
            for s2 in second_set:
                # Check if all symbols in s1 and s2 are empty (epsilon).
                s1_empty = all(c.is_empty() for c in s1)
                s2_empty = all(c.is_empty() for c in s2)
                # If both strings are empty, add an epsilon to the result set.
                if s1_empty and s2_empty:
                    result.add((self.grammar.get_epsilon(),))
                    continue
                # If only the first string is empty, add the second string to the result set.
                elif s1_empty:
                    result.add(s2)
                    continue
                # If only the second string is empty, add the first string to the result set.
                elif s2_empty:
                    result.add(s1)
                    continue
                # Concatenate the two strings and add the result to the result set.
                result.add((s1 + s2)[:k])
        return result

    def get_possible_strings(self, rule, k, prev_first_k):
        possible_strings = []
        queue = deque()
        # Start with the original rule.
        queue.append(list(rule))
        while queue:
            # Take the first item from the queue for processing.
            current_rule = queue.popleft()
            # Check if the first k symbols are all terminals or empty.
            if all(isinstance(c, Terminal) for c in current_rule[:k]):
                if all(nt_c.is_empty() for nt_c in current_rule[:k]):
                    # If all symbols are empty, add epsilon to possible strings.
                    possible_strings.append((self.grammar.get_epsilon(),))
                else:
                    # Add the first k terminals as a possible string.
                    possible_strings.append(tuple(current_rule[:k]))
                continue
            # Iterate through each symbol in the current rule.
            for i, c in enumerate(current_rule):
                # Process only non-terminal symbols.
                if isinstance(c, NonTerminal):
                    # For each possible first set of the non-terminal, create a new rule variant.
                    for nt_first in prev_first_k[c]:
                        new_rule = current_rule.copy()
                        # Check if all symbols in the first set of non-terminal are empty.
                        is_prev_first_empty = all(nt_c.is_empty() for nt_c in nt_first)
                        # Handle empty symbols in non-terminal expansions.
                        if is_prev_first_empty and len(current_rule) > 1:
                            # If the first set is empty and rule is not a single symbol, remove the non-terminal.
                            new_rule[i:i + 1] = []
                        else:
                            # Replace the non-terminal with its first set.
                            new_rule[i:i + 1] = nt_first
                        # Add the new rule variant to the queue for further processing.
                        queue.append(new_rule)
                    break
        return possible_strings

    @staticmethod
    def tuples_to_strings(table):
        result = {}
        for key, value in table.items():
            result[key] = set()
            for v in value:
                result[key].add(''.join([str(c) for c in v]))
        return result


class ConstructParsingTable:
    def __init__(self, grammar, first_sets, follow_sets):
        self.grammar = grammar
        self.first_sets = first_sets
        self.follow_sets = follow_sets
        self.parsing_table = {}

    def construct(self):
        # Construct the parsing table for each non-terminal and its productions.
        for non_terminal, productions in self.grammar.rules.items():
            for production in productions:
                # Find the first set for the production.
                first_production = self._find_first_production(production)
                # Add rules to the parsing table based on the first set.
                for terminal_tuple in first_production:
                    terminal = terminal_tuple[0]
                    if terminal != Terminal('ε'):
                        self._add_to_parsing_table(non_terminal, terminal, production)
                # Process productions that can derive epsilon.
                if (Terminal('ε'),) in first_production:
                    self._process_epsilon(non_terminal, production)

        return self.parsing_table

    def _find_first_production(self, production):
        first_production = set()
        # Convert the production into a list of symbol texts.
        production = [c.text for c in Grammar.get_tnt_string(self.grammar, production)]

        for t_symbol in production:
            # Determine if the symbol is a terminal or non-terminal.
            symbol = Terminal(t_symbol) if Terminal(t_symbol) in self.grammar.terminals else NonTerminal(t_symbol)
            # Get the first set of the symbol.
            first_symbol = self.first_sets[symbol] if symbol in self.first_sets else {(symbol,)}

            if (Terminal('ε'),) not in first_symbol:
                first_production.update(first_symbol)
            # Stop if the symbol cannot derive epsilon.
            if (Terminal('ε'),) not in first_symbol:
                break
        else:
            # Add epsilon if all symbols in the production can derive epsilon.
            first_production.add((Terminal('ε'),))
        return first_production

    def _add_to_parsing_table(self, non_terminal, terminal, production):
        # Add a rule to the parsing table.
        production = [c.text for c in Grammar.get_tnt_string(self.grammar, production)]
        # Check for conflicts in the parsing table.
        if (non_terminal, terminal) not in self.parsing_table:
            self.parsing_table[(non_terminal, terminal)] = production
        else:
            raise ValueError(f"Grammar is not LL(1): Conflict at ({non_terminal}, {terminal})")

    def _process_epsilon(self, non_terminal, production):
        # Process productions that can derive epsilon.
        production = [c.text for c in Grammar.get_tnt_string(self.grammar, production)]
        # Use the follow set of the non-terminal to add rules for deriving epsilon.
        for terminal in self.follow_sets[NonTerminal(non_terminal)]:
            if (non_terminal, terminal[0]) not in self.parsing_table:
                self.parsing_table[(non_terminal, terminal[0])] = production
            else:
                raise ValueError(f"Grammar is not LL(1): Conflict at ({non_terminal}, {terminal})")

def parse_with_control_table(grammar, expression):
    first_follow = FirstFollow(grammar)
    first_k = first_follow.compute_first_k(1)
    follow_k = first_follow.compute_follow_k(1, first_k)
    parser = ConstructParsingTable(grammar, first_k, follow_k)
    table = parser.construct()
    print("Parser control table:")
    for key, value in table.items():
        print(str(key) + ":")
        joined_value = ''.join(map(str, value))
        print(f"{str(key[0])} -> {'ε' if not joined_value else joined_value}")
        print("")


def view_result(grammar, expression):
    first_follow = FirstFollow(grammar)
    first_k = first_follow.compute_first_k(2)
    follow_k = first_follow.compute_follow_k(1, first_k)
    print("Terminals:")
    print(grammar.terminals)
    print("Non-Terminals:")
    print(grammar.non_terminals)
    print("Epsilon-Producers:")
    print(grammar.epsilon_producers)
    print("First(k):")
    for n in grammar.non_terminals:
        print(str(n) + ":")
        print(', '.join(''.join(map(str, tupl)) for tupl in first_k[n]))
    print("Follow(k):")
    for n in grammar.non_terminals:
        print(str(n) + ":")
        print(', '.join(''.join(map(str, tupl)) for tupl in follow_k[n]))
    parse_with_control_table(grammar, expression)


if __name__ == "__main__":
    example_grammar = Grammar(Grammar.read_grammar_from_file("in.txt"))
    example_expression = ""
    view_result(example_grammar, example_expression)
