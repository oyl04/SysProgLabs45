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
        # A list to hold production productions for this non-terminal.
        self.productions = []

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return self.text

    def __eq__(self, o: object) -> bool:
        return o.text == self.text

    def __hash__(self) -> int:
        return hash(self.text)


class Grammar:

    def __init__(self, productions):
        self.pattern = None
        self.terminals = []
        self.non_terminals = []
        # Process the given productions to populate terminals and non-terminals.
        self.get_productions(productions)
        self.productions = productions
        # Identify non-terminals that can produce an epsilon (empty string).
        self.nullable_non_terminals = self.find_epsilon_producing_non_terminals()

    def get_epsilon(self):
        # Find a terminal representing an epsilon.
        eps = next((t for t in self.terminals if t.is_empty()), None)
        if eps is None:
            eps = Terminal('eps')
        return eps

    @staticmethod
    # Generate a regex pattern to identify terminals and non-terminals in text.
    def get_pattern(text_set):
        return f"({'|'.join(sorted([re.escape(t.text) for t in text_set], key=lambda x: len(x), reverse=True))})"

    # Process the given productions to populate terminals and non-terminals.
    def get_productions(self, productions):
        # Add non-terminals defined in the productions to the grammar's non_terminals list.
        self.non_terminals += [NonTerminal(n) for n in list(productions.keys())]
        # Create a regular expression pattern to identify non-terminals.
        n_pattern = Grammar.get_pattern(self.non_terminals)

        terminals = set()
        for n in self.non_terminals:
            for nt_production in productions[n.text]:
                # Split each rule production and identify terminals.
                g = nt_production.split()
                # Add identified terminals to the set.
                t = [re.sub(n_pattern, ' ', p).split() for p in g]
                terminals |= set(itertools.chain.from_iterable(t))
        # Create Terminal objects for each identified terminal symbol.
        self.terminals = [Terminal(term) for term in terminals]
        # Print the identified terminals and non-terminals.
        print(self.terminals, self.non_terminals)

        # Create a regular expression pattern to identify terminals.
        t_pattern = Grammar.get_pattern(self.terminals)

        # Combine the terminal and non-terminal patterns into a single pattern.
        nt_pattern = f"{n_pattern}|{t_pattern}"
        self.pattern = nt_pattern

        # Process each production for each non-terminal and store in their 'productions' attribute.
        for n in self.non_terminals:
            for nt_production in productions[n.text]:
                n.productions.append([])
                for m in re.finditer(nt_pattern, nt_production):
                    # Depending on whether the match is a terminal or non-terminal, add it to the production.
                    if m.group(1):
                        n.productions[-1].append(next((nt for nt in self.non_terminals if nt.text == m.group(1))))
                    elif m.group(2):
                        n.productions[-1].append(next((t for t in self.terminals if t.text == m.group(2))))

    def find_epsilon_producing_non_terminals(self):
        nullable_non_terminals = set()
        # Check each non-terminal's production productions.
        for nt in self.non_terminals:
            for production in nt.productions:
                # If any production of a non-terminal directly produces an epsilon, add it to the set.
                if any(isinstance(symbol, Terminal) and symbol.is_empty() for symbol in production):
                    nullable_non_terminals.add(nt)
                    break

        # Continuously check for non-terminals that can indirectly produce an epsilon.
        changed = True
        while changed:
            changed = False
            for nt in self.non_terminals:
                if nt in nullable_non_terminals:
                    continue
                for production in nt.productions:
                    # If all symbols in a production are epsilon producers or empty terminals, add the non-terminal.
                    if all(symbol in nullable_non_terminals or (isinstance(symbol, Terminal) and symbol.is_empty()) for
                           symbol in production):
                        nullable_non_terminals.add(nt)
                        changed = True
                        break

        return nullable_non_terminals

    # Method to read grammar productions from a file.
    @staticmethod
    def read_grammar_from_file(file_path):
        productions = {}

        with open(file_path, 'r') as file:
            for line in file:
                # Split each line at the '->' symbol to separate the left and right parts of the rule production.
                parts = line.strip().split('->')
                if len(parts) == 2:
                    left, right = parts
                    # Trim whitespace from the left-hand side (non-terminal) of the production.
                    left = left.strip()
                    # Split the right-hand side of the rule into individual productions using '|' as the delimiter.
                    right_productions = right.strip().split("|")
                    # If the non-terminal is not already in the productions dictionary, add it with its productions.
                    # Otherwise, extend its list of productions with the new ones.
                    if left not in productions:
                        productions[left] = right_productions
                    else:
                        productions[left].extend(right_productions)

        for key in productions.keys():
            productions[key] = ['ε' if prod == 'epsilon' else prod for prod in productions[key]]

        return productions

    def get_tnt_string(self, text):
        result = []
        for m in re.finditer(self.pattern, text):
            if m.group(1):
                # Append the corresponding non-terminal object to the result
                result.append(next((nt for nt in self.non_terminals if nt.text == m.group(1))))
            elif m.group(2):
                # Append the corresponding terminal object to the result
                result.append(next((t for t in self.terminals if t.text == m.group(2))))
        return result

    def build_ast(self, rule_sequence):
        root = ASTNode(self.non_terminals[0])
        stack = [root]
        for rule in rule_sequence:
            # Unpack the rule into non-terminal, token, and production.
            nt, _, prod = rule
            nt = nt[0]
            curr_node = stack.pop()
            if curr_node.symbol == nt:
                # Iterate over the production in reverse order.
                # This is because we want to process children from left to right, and the stack is LIFO.
                for c in prod[::-1]:
                    # For each symbol in the production, create a new AST node and add it as a child.
                    if isinstance(c, Terminal):
                        curr_node.add_child(ASTNode(c))
                    else:
                        curr_node.add_child(ASTNode(c))
                        # If the child is a non-terminal, push it onto the stack for further processing.
                        stack.append(curr_node.children[-1])
        return root


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
                for production in n.productions:
                    # Calculate possible strings of length k or less from the production using the current first set.
                    possible_strings = self.get_possible_strings(production, k, first)
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
        seen_non_terminals = set()
        # Add the start symbol to the seen non-terminals set.
        seen_non_terminals.add(self.grammar.non_terminals[0])

        # Loop until the follow set no longer changes between iterations.
        while follow != prev_follow:
            prev_follow = follow.copy()
            for key, value in prev_follow.items():
                # Create a value copy of each set in the dictionary to
                # avoid modifying the original sets during iteration.
                prev_follow[key] = value.copy()

            # Store newly seen non-terminals during this iteration.
            new_seen_non_terminals = []
            for nt in seen_non_terminals:
                for production in nt.productions:
                    for i, c in enumerate(production):
                        # Only process non-terminal symbols.
                        if isinstance(c, NonTerminal):
                            # Add the non-terminal to the buffer of newly seen non-terminals.
                            new_seen_non_terminals.append(c)
                            # Get the symbols following the current non-terminal in the production.
                            after = production[i + 1:]
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

            seen_non_terminals |= set(new_seen_non_terminals)

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

    def get_possible_strings(self, production, k, prev_first_k):
        possible_strings = []
        queue = deque()
        # Start with the original production.
        queue.append(list(production))
        while queue:
            # Take the first item from the queue for processing.
            current_production = queue.popleft()
            # Check if the first k symbols are all terminals or empty.
            if all(isinstance(c, Terminal) for c in current_production[:k]):
                if all(nt_c.is_empty() for nt_c in current_production[:k]):
                    # If all symbols are empty, add epsilon to possible strings.
                    possible_strings.append((self.grammar.get_epsilon(),))
                else:
                    # Add the first k terminals as a possible string.
                    possible_strings.append(tuple(current_production[:k]))
                continue
            # Iterate through each symbol in the current production.
            for i, c in enumerate(current_production):
                # Process only non-terminal symbols.
                if isinstance(c, NonTerminal):
                    # For each possible first set of the non-terminal, create a new production variant.
                    for nt_first in prev_first_k[c]:
                        new_production = current_production.copy()
                        # Check if all symbols in the first set of non-terminal are empty.
                        is_prev_first_empty = all(nt_c.is_empty() for nt_c in nt_first)
                        # Handle empty symbols in non-terminal expansions.
                        if is_prev_first_empty and len(current_production) > 1:
                            # If the first set is empty and production is not a single symbol, remove the non-terminal.
                            new_production[i:i + 1] = []
                        else:
                            # Replace the non-terminal with its first set.
                            new_production[i:i + 1] = nt_first
                        # Add the new production variant to the queue for further processing.
                        queue.append(new_production)
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
        for non_terminal, productions in self.grammar.productions.items():
            for production in productions:
                # Find the first set for the production.
                first_production = self._find_first_production(production)
                # Add productions to the parsing table based on the first set.
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
        # Add a production to the parsing table.
        production = [c.text for c in Grammar.get_tnt_string(self.grammar, production)]
        # Check for conflicts in the parsing table.
        if (non_terminal, terminal) not in self.parsing_table:
            self.parsing_table[(non_terminal, terminal)] = production
        else:
            raise ValueError(f"Grammar is not LL(1): Conflict at ({non_terminal}, {terminal})")

    def _process_epsilon(self, non_terminal, production):
        # Process productions that can derive epsilon.
        production = [c.text for c in Grammar.get_tnt_string(self.grammar, production)]
        # Use the follow set of the non-terminal to add productions for deriving epsilon.
        for terminal in self.follow_sets[NonTerminal(non_terminal)]:
            if (non_terminal, terminal[0]) not in self.parsing_table:
                self.parsing_table[(non_terminal, terminal[0])] = production
            else:
                raise ValueError(f"Grammar is not LL(1): Conflict at ({non_terminal}, {terminal})")


class LLkAnalyzer:
    def __init__(self, parsing_table, grammar):
        self.parsing_table = parsing_table
        self.grammar = grammar
        self.start_symbol = self.grammar.non_terminals[0]

    def str_to_sym(self, sym):
        if Terminal(sym) in self.grammar.terminals:
            return Terminal(sym)
        else:
            return NonTerminal(sym)

    def parse(self, tokens):
        tokens.append('$')  # Append end marker to the token list
        applied_productions = []
        stack = ['$', self.start_symbol]  # Initialize stack with end marker and start symbol

        current_token_index = 0
        step = 0
        while len(stack) > 0:
            print(f"Step #{step}")
            step += 1
            print(list(reversed(stack)))
            print(tokens[current_token_index:])
            top = stack.pop()
            # Prints for current stack, top element, and remaining tokens

            if isinstance(top, Terminal):
                #  Match terminal symbol with current token
                if top == tokens[current_token_index]:
                    current_token_index += 1
                else:
                    raise SyntaxError(f"Unexpected token: Expected {top}, found {tokens[current_token_index]}")
            elif isinstance(top, NonTerminal):
                #  Process non-terminal
                #  Special processing when the current token is the end marker '$'
                if isinstance(tokens[current_token_index], str) and tokens[current_token_index] == '$':
                    #  Look up in the parsing table for an entry with the non-terminal and epsilon
                    entry = self.parsing_table.get((str(top), Terminal('ε')))
                    #  If an entry is found, it means the non-terminal can be replaced with epsilon
                    if entry is not None:
                        applied_productions.append((str(top), Terminal('ε'), entry))
                        for symbol in reversed(entry):
                            if symbol == 'ε':
                                continue
                            stack.append(self.str_to_sym(symbol))
                        continue
                # If an entry is found in the parsing table, it means there is a production that can be applied
                # for the current non-terminal (top of the stack) and the current input token.
                entry = self.parsing_table.get((str(top), tokens[current_token_index]))
                if entry is not None:
                    applied_productions.append((str(top), tokens[current_token_index], entry))
                    # Iterate through the symbols in the found production in reverse order.
                    for symbol in reversed(entry):
                        # If the symbol is epsilon -> we do not need to match any input token,
                        # so we can skip it.
                        if symbol == 'ε':
                            continue
                        # For other symbols, convert them to terminals or non-terminals and push them onto the stack.
                        stack.append(self.str_to_sym(symbol))
                else:
                    # If no entry is found in the parsing table, it means there is no valid production for the current
                    raise SyntaxError(f"No production to parse: {top} with token {tokens[current_token_index]}")
            # This condition checks if we have reached the end of the stack.
            elif top == '$':
                if tokens[current_token_index] == '$':
                    print("Parsing successful!")
                # If the current token is also the end marker '$', it means the input string
                # has been successfully parsed according to the grammar productions.
                else:
                    raise SyntaxError("Unexpected end of input")

            else:
                raise ValueError(f"Invalid symbol on stack: {top}")

        if current_token_index < len(tokens) - 1:
            raise SyntaxError("Input not fully parsed")

        terms = []
        #  Convert applied productions to terminals and non-terminals
        for production in applied_productions:
            func = lambda x: list(map(self.str_to_sym, x))
            terms.append((func(production[0]), production[1], func(production[2])))
        return terms


class RecursiveDescentParser:
    def __init__(self, grammar):
        self.grammar = grammar
        self.tokens = []
        self.index = 0

    def parse(self, input_str):
        # Convert input string into tokens based on the grammar
        self.tokens = [c.text for c in self.grammar.get_tnt_string(input_str)]
        self.index = 0
        # Start symbol is the first non-terminal in grammar
        start_symbol = self.grammar.non_terminals[0]
        # Check if the entire token list can be parsed from the start symbol
        return self.parse_non_terminal(start_symbol) and self.index == len(self.tokens)

    def parse_non_terminal(self, non_terminal):
        save_index = self.index
        # Try each production of the non-terminal to see if it matches the tokens
        for production in non_terminal.productions:
            self.index = save_index  # Reset index to try next production
            if all(self.parse_symbol(symbol) for symbol in production):
                return True  # Successful parsing of this non-terminal
        self.index = save_index
        return False  # This non-terminal can't be parsed

    def parse_symbol(self, symbol):
        if isinstance(symbol, Terminal):
            if symbol.is_empty():
                return True
            # Match terminal with current token
            return self.match(symbol.text)
        elif isinstance(symbol, NonTerminal):
            # Recursive parsing for non-terminal
            return self.parse_non_terminal(symbol)
        # If symbol type is unrecognized, return False
        return False

    def match(self, terminal):
        # Check if the current token matches the terminal
        if self.index < len(self.tokens) and self.tokens[self.index] == terminal:
            self.index += 1
            return True  # Successful match
        return False


class ASTNode:
    def __init__(self, symbol):
        self.symbol = symbol
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    @staticmethod
    def print_ast(node, prefix="", last=True):
        # Determine the branching symbol ('└── ' for the last child, '├── ' otherwise)
        turn = '└── ' if last else '├── '
        # Print the current node's symbol with the appropriate prefix and branch symbol
        print(prefix + turn + str(node.symbol))
        # If this is the last child, add whitespace; otherwise, add a vertical line.
        prefix += '    ' if last else '│   '

        # Count the number of children of the current node
        child_count = len(node.children)
        for i, child in enumerate(node.children):
            # Determine if the current child is the last in the list of children
            is_last = i == (child_count - 1)
            # Recursively call print_ast for each child, updating the prefix and last flag
            ASTNode.print_ast(child, prefix, last=is_last)


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
    recursive_parser = RecursiveDescentParser(grammar)
    print(f'Recursive Descent Parsing "{expression}": {recursive_parser.parse(expression)}')
    print("Analyzer process:")
    applied_productions = None
    try:
        analyzer = LLkAnalyzer(table, grammar)
        tokenized = grammar.get_tnt_string(expression)
        applied_productions = analyzer.parse(tokenized)
    except SyntaxError as se:
        print(f"Got an error: {str(se)}")
    except ValueError as ve:
        print(f"Got an error: {str(ve)}")
    return applied_productions


def view_result(grammar, expression):
    first_follow = FirstFollow(grammar)
    first_k = first_follow.compute_first_k(1)
    follow_k = first_follow.compute_follow_k(1, first_k)
    print("Terminals:")
    print(grammar.terminals)
    print("Non-Terminals:")
    print(grammar.non_terminals)
    print("Epsilon-Producers:")
    print(grammar.nullable_non_terminals)
    print("First(k):")
    for n in grammar.non_terminals:
        print(str(n) + ":")
        print(', '.join(''.join(map(str, sym)) for sym in first_k[n]))
    print("Follow(k):")
    for n in grammar.non_terminals:
        print(str(n) + ":")
        print(', '.join(''.join(map(str, sym)) for sym in follow_k[n]))
    order_of_rules = parse_with_control_table(grammar, expression)
    if order_of_rules:
        print("Applied Rules:")
        for i, rule in enumerate(order_of_rules):
            print(f"{rule[0]} -> {rule[2]}")
        print("Abstract Syntax Tree:")
        root = grammar.build_ast(order_of_rules)
        ASTNode.print_ast(root)
    else:
        print("Impossible to build AST")


if __name__ == "__main__":
    example_grammar = Grammar(Grammar.read_grammar_from_file("in.txt"))
    example_expression = "(a+a)*a"
    view_result(example_grammar, example_expression)
