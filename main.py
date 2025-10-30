import pprint

class SLRParser:
    def __init__(self, grammar_str):
        self.grammar = self._parse_grammar(grammar_str)
        self.augmented_grammar = self._augment_grammar(self.grammar)
        self.terminals = self._get_terminals()
        self.non_terminals = self._get_non_terminals()
        
        # These will be computed later
        self.first_sets = {}
        self.follow_sets = {}
        self.canonical_collection = []
        self.action_table = {}
        self.goto_table = {}

    def _parse_grammar(self, grammar_str):
        """Parses the grammar string into a list of productions."""
        grammar = []
        for line in grammar_str.strip().split('\n'):
            head, body = line.split('::=')
            # Productions are stored as (head, tuple(body_symbols))
            grammar.append((head.strip(), tuple(body.strip().split())))
        return grammar

    def _augment_grammar(self, grammar):
        """Adds a new start symbol S' -> S $."""
        start_symbol = grammar[0][0]
        # The document uses S ::= E $, so the start symbol is 'S' and '$' is already there.
        # A more general approach adds S' -> S. We will follow the general approach.
        # Let's call the new start symbol S_AUG.
        augmented_start = f"{start_symbol}'"
        return [(augmented_start, (start_symbol,))] + grammar

    def _get_non_terminals(self):
        """Extracts non-terminals from the grammar."""
        return sorted(list(set(p[0] for p in self.augmented_grammar)))

    def _get_terminals(self):
        """Extracts terminals from the grammar."""
        non_terminals = self._get_non_terminals()
        terminals = set()
        for _, body in self.augmented_grammar:
            for symbol in body:
                if symbol not in non_terminals:
                    terminals.add(symbol)
        # Add the end-of-input marker, which is crucial for parsing
        terminals.add('$')
        return sorted(list(terminals))

    def print_grammar_info(self):
        print("--- Augmented Grammar ---")
        for i, (head, body) in enumerate(self.augmented_grammar):
             print(f"{i}: {head} -> {' '.join(body)}")
        print("\n--- Terminals ---")
        print(self.terminals)
        print("\n--- Non-Terminals ---")
        print(self.non_terminals)


class SLRParser(SLRParser): # Extending the previous class
    def _compute_first_sets(self):
        """Computes the FIRST sets for all symbols in the grammar."""
        first = {symbol: set() for symbol in self.non_terminals + self.terminals}
        
        # Terminals' FIRST set is the terminal itself
        for t in self.terminals:
            first[t] = {t}

        while True:
            updated = False
            for head, body in self.augmented_grammar:
                # For each production A -> B1 B2 ...
                for symbol in body:
                    # first(A) includes first(B1)
                    for f in first[symbol]:
                        if f not in first[head]:
                            first[head].add(f)
                            updated = True
                    # If B1 can be empty (epsilon), we also need to consider B2, etc.
                    # This grammar doesn't have epsilon, so we break.
                    break # Simplified for non-epsilon grammar
            if not updated:
                break
        self.first_sets = first

    def _compute_follow_sets(self):
        """Computes the FOLLOW sets for all non-terminals."""
        follow = {nt: set() for nt in self.non_terminals}
        start_symbol = self.augmented_grammar[0][0]
        follow[start_symbol].add('$')

        while True:
            updated = False
            for head, body in self.augmented_grammar:
                # For each production A -> ... B C ...
                for i in range(len(body)):
                    B = body[i]
                    if B in self.non_terminals:
                        # Find FOLLOW(B)
                        # Case 1: A -> ... B C ...
                        # Add FIRST(C) to FOLLOW(B)
                        if i + 1 < len(body):
                            beta = body[i+1]
                            for f in self.first_sets[beta]:
                                if f not in follow[B]:
                                    follow[B].add(f)
                                    updated = True
                        # Case 2: A -> ... B
                        # Add FOLLOW(A) to FOLLOW(B)
                        else:
                            for f in follow[head]:
                                if f not in follow[B]:
                                    follow[B].add(f)
                                    updated = True
            if not updated:
                break
        self.follow_sets = follow

class SLRParser(SLRParser): # Extending the class
    def _closure(self, items):
        """Computes the closure of a set of LR(0) items."""
        closure_set = set(items)
        while True:
            new_items = set()
            for head, body, dot_pos in closure_set:
                if dot_pos < len(body):
                    symbol_after_dot = body[dot_pos]
                    if symbol_after_dot in self.non_terminals:
                        for prod_head, prod_body in self.augmented_grammar:
                            if prod_head == symbol_after_dot:
                                new_item = (prod_head, prod_body, 0)
                                if new_item not in closure_set:
                                    new_items.add(new_item)
            if not new_items:
                break
            closure_set.update(new_items)
        return frozenset(closure_set)

    def _goto(self, item_set, symbol):
        """Computes the GOTO set for a given item set and symbol."""
        next_items = set()
        for head, body, dot_pos in item_set:
            if dot_pos < len(body) and body[dot_pos] == symbol:
                next_items.add((head, body, dot_pos + 1))
        return self._closure(next_items)

    def _build_canonical_collection(self):
        """Builds the canonical collection of LR(0) item sets (the DFA)."""
        # Initial state I0
        initial_prod = self.augmented_grammar[0]
        initial_item = (initial_prod[0], initial_prod[1], 0)
        I0 = self._closure({initial_item})
        
        collection = [I0]
        # Map frozenset to state index for quick lookup
        state_map = {I0: 0}
        
        worklist = [I0]
        while worklist:
            current_set = worklist.pop(0)
            current_state_idx = state_map[current_set]
            
            all_symbols = self.terminals + self.non_terminals
            for symbol in all_symbols:
                if symbol == '$' and self.augmented_grammar[0][0].endswith("'"): # Exclude '$' from goto unless it's part of a rule
                    continue

                next_set = self._goto(current_set, symbol)
                if not next_set:
                    continue
                
                if next_set not in state_map:
                    state_map[next_set] = len(collection)
                    collection.append(next_set)
                    worklist.append(next_set)
                
                # Record the transition (for the GOTO table later)
                next_state_idx = state_map[next_set]
                if current_state_idx not in self.goto_table:
                    self.goto_table[current_state_idx] = {}
                self.goto_table[current_state_idx][symbol] = next_state_idx
                
        self.canonical_collection = collection


class SLRParser(SLRParser): # Extending the class
    def _build_parsing_table(self):
        """Builds the SLR ACTION and GOTO tables."""
        # Initialize tables
        for i in range(len(self.canonical_collection)):
            self.action_table[i] = {}

        # Rule 1: Shift actions
        for i, transitions in self.goto_table.items():
            for symbol, j in transitions.items():
                if symbol in self.terminals:
                    self.action_table[i][symbol] = f's{j}'

        # Production map for reduce actions
        prod_map = {(p[0], p[1]): i for i, p in enumerate(self.augmented_grammar)}

        # Rules 2 & 3: Reduce and Accept actions
        for i, item_set in enumerate(self.canonical_collection):
            for head, body, dot_pos in item_set:
                # If dot is at the end, it's a reduce or accept item
                if dot_pos == len(body):
                    # Rule 3: Accept
                    if head == self.augmented_grammar[0][0]: # e.g., S' -> S .
                        self.action_table[i]['$'] = 'accept'
                    # Rule 2: Reduce
                    else:
                        prod_num = prod_map[(head, body)]
                        for symbol in self.follow_sets[head]:
                            if symbol in self.action_table[i]:
                                # Conflict detected!
                                print(f"CONFLICT in state {i} on symbol '{symbol}':")
                                print(f"  Existing: {self.action_table[i][symbol]}")
                                print(f"  New: r{prod_num}")
                                print(f"  Grammar is not SLR(1).")
                            else:
                                self.action_table[i][symbol] = f'r{prod_num}'


class SLRParser(SLRParser): # Extending the class
    def parse(self, input_string):
        """Parses an input string using the generated SLR table."""
        tokens = input_string.split() + ['$']
        stack = [0]
        
        print("\n--- Parsing Trace ---")
        print(f"{'Stack':<30} | {'Input':<30} | {'Action'}")
        print("-" * 75)

        ip = 0 # Input pointer
        while True:
            state = stack[-1]
            token = tokens[ip]
            
            stack_str = ' '.join(map(str, stack))
            input_str = ' '.join(tokens[ip:])
            
            if token not in self.action_table[state]:
                action = "Error: No action"
                print(f"{stack_str:<30} | {input_str:<30} | {action}")
                print("\nParsing failed: Syntax Error.")
                return False

            action = self.action_table[state][token]
            print(f"{stack_str:<30} | {input_str:<30} | {action}")

            if action.startswith('s'): # Shift
                next_state = int(action[1:])
                stack.append(token)
                stack.append(next_state)
                ip += 1
            elif action.startswith('r'): # Reduce
                prod_num = int(action[1:])
                head, body = self.augmented_grammar[prod_num]
                # Pop 2 * |body| items from stack (symbol and state)
                for _ in range(2 * len(body)):
                    stack.pop()
                
                # New state on top of stack
                prev_state = stack[-1]
                # Push the non-terminal and the GOTO state
                stack.append(head)
                stack.append(self.goto_table[prev_state][head])
            elif action == 'accept':
                print("\nParsing successful: String accepted.")
                return True
            else:
                print(f"\nParsing failed: Unknown action '{action}'.")
                return False

    def build(self):
        """A helper method to run all build steps."""
        self._compute_first_sets()
        self._compute_follow_sets()
        self._build_canonical_collection()
        self._build_parsing_table()

# Main execution block

# Grammar from the assignment (rules 1-5, ignoring the S ::= E $ for now)
# Let's use a standard expression grammar and augment it.
grammar_text = """
E ::= E + T
E ::= T
T ::= id
T ::= ( E )
"""

# Let's adjust it to match the document's production numbering.
# We will augment it correctly with S' -> S. Let S be the start symbol.
# S -> E
# E -> E + T
# E -> T
# T -> id
# T -> ( E )
grammar_from_doc = """
S ::= E
E ::= E + T
E ::= T
T ::= id
T ::= ( E )
"""

# Create and build the parser
parser = SLRParser(grammar_from_doc)
parser.build()

# --- Print all the generated information for verification ---

# Print Grammar Info
parser.print_grammar_info()

# Print FIRST and FOLLOW sets
print("\n--- FIRST Sets ---")
pprint.pprint(parser.first_sets)
print("\n--- FOLLOW Sets ---")
pprint.pprint(parser.follow_sets)

# Print the Canonical Collection (DFA States)
print("\n--- Canonical Collection (DFA States) ---")
for i, item_set in enumerate(parser.canonical_collection):
    print(f"I{i}:")
    for head, body, dot_pos in item_set:
        body_str = list(body)
        body_str.insert(dot_pos, '.')
        print(f"  {head} -> {' '.join(body_str)}")
    print("-" * 10)

# Print the Parsing Table
print("\n--- SLR Parsing Table ---")
header = ['STATE'] + parser.terminals + parser.non_terminals
print(f"{header[0]:<6}" + "".join([f"{h:<8}" for h in header[1:]]))
print("-" * (6 + 8 * (len(header) - 1)))
for i in range(len(parser.canonical_collection)):
    row = [f"{i:<6}"]
    # Action part
    for t in parser.terminals:
        row.append(f"{parser.action_table[i].get(t, ''):<8}")
    # Goto part
    for nt in parser.non_terminals:
         # Note: GOTO table is built slightly differently, need to access it correctly
        goto_state = parser.goto_table.get(i, {}).get(nt, '')
        row.append(f"{str(goto_state):<8}")
    print("".join(row))


# --- Test the parser with an input string ---
input_to_parse = "id + id" 
parser.parse(input_to_parse)

print("\n" + "="*80 + "\n")

input_to_parse_2 = "( id + id )"
parser.parse(input_to_parse_2)