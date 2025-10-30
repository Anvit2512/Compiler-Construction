# UCS 802 Compiler Construction: Lab Assignment II
# SLR Parser for the grammar: E -> E+T | T, T -> T*F | F, F -> (E) | id

class SLRParser:
    def __init__(self, grammar_str):
        # --- Initialization ---
        self.grammar = {}
        self.terminals = set()
        self.non_terminals = set()
        self.start_symbol = ""
        self.augmented_start_symbol = ""
        self.productions = []
        self.first_sets = {}
        self.follow_sets = {}
        self.canonical_collection = []
        self.goto_map = {}
        self.action_table = {}
        self.goto_table = {}

        # --- Core Logic Execution ---
        self._parse_grammar(grammar_str)
        self._compute_first_and_follow()
        self._build_canonical_collection()
        self._build_parsing_table()

    def _parse_grammar(self, grammar_str):
        """Parses the grammar string, augments it, and identifies symbols."""
        lines = grammar_str.strip().split('\n')
        self.start_symbol = lines[0].split('->')[0].strip()
        self.augmented_start_symbol = self.start_symbol + "'"
        
        # Augment the grammar with E' -> E
        self.grammar[self.augmented_start_symbol] = [[self.start_symbol]]
        # Production 0: E' -> E
        self.productions.append((self.augmented_start_symbol, (self.start_symbol,)))

        for line in lines:
            head, body_str = line.split('->')
            head = head.strip()
            self.non_terminals.add(head)
            if head not in self.grammar:
                self.grammar[head] = []
            
            prods = [p.strip().split() for p in body_str.split('|')]
            for prod_body in prods:
                self.grammar[head].append(tuple(prod_body))
                self.productions.append((head, tuple(prod_body)))
                for symbol in prod_body:
                    if not symbol[0].isupper(): # Terminals don't start with uppercase
                        self.terminals.add(symbol)
        
        self.non_terminals.add(self.augmented_start_symbol)
        self.terminals.add('$')

    def _compute_first_and_follow(self):
        """Computes FIRST and FOLLOW sets for all non-terminals."""
        # Initialize FIRST sets
        for nt in self.non_terminals:
            self.first_sets[nt] = set()
        
        # Iteratively compute FIRST sets
        while True:
            updated = False
            for head, bodies in self.grammar.items():
                for body in bodies:
                    for symbol in body:
                        original_size = len(self.first_sets[head])
                        if symbol in self.terminals:
                            self.first_sets[head].add(symbol)
                            break
                        else: # Non-terminal
                            self.first_sets[head].update(self.first_sets[symbol])
                            if 'epsilon' not in self.first_sets[symbol]:
                                break
                    if len(self.first_sets[head]) > original_size: updated = True
            if not updated: break

        # Initialize FOLLOW sets
        for nt in self.non_terminals:
            self.follow_sets[nt] = set()
        self.follow_sets[self.start_symbol].add('$')

        # Iteratively compute FOLLOW sets
        while True:
            updated = False
            for head, bodies in self.grammar.items():
                for body in bodies:
                    for i, symbol in enumerate(body):
                        if symbol in self.non_terminals:
                            original_size = len(self.follow_sets[symbol])
                            # Check symbols that follow
                            trailer = body[i+1:]
                            if trailer:
                                first_of_trailer = set()
                                for t_symbol in trailer:
                                    if t_symbol in self.terminals:
                                        first_of_trailer.add(t_symbol)
                                        break
                                    else:
                                        first_of_trailer.update(self.first_sets[t_symbol])
                                        if 'epsilon' not in self.first_sets[t_symbol]:
                                            break
                                self.follow_sets[symbol].update(first_of_trailer)
                            else: # Nothing follows, so FOLLOW(symbol) includes FOLLOW(head)
                                self.follow_sets[symbol].update(self.follow_sets[head])
                            if len(self.follow_sets[symbol]) > original_size: updated = True
            if not updated: break

    def _closure(self, items):
        """Computes the closure of a set of LR(0) items."""
        closure_set = set(items)
        worklist = list(items)
        while worklist:
            head, body, dot_pos = worklist.pop(0)
            if dot_pos < len(body):
                symbol_after_dot = body[dot_pos]
                if symbol_after_dot in self.non_terminals:
                    for prod_body in self.grammar.get(symbol_after_dot, []):
                        new_item = (symbol_after_dot, tuple(prod_body), 0)
                        if new_item not in closure_set:
                            closure_set.add(new_item)
                            worklist.append(new_item)
        return frozenset(closure_set)

    def _goto(self, items, symbol):
        """Computes the GOTO set for a set of items and a grammar symbol."""
        new_items = set()
        for head, body, dot_pos in items:
            if dot_pos < len(body) and body[dot_pos] == symbol:
                new_items.add((head, body, dot_pos + 1))
        return self._closure(new_items)

    def _build_canonical_collection(self):
        """TASK 1: Builds the canonical collection of LR(0) items."""
        initial_item = (self.augmented_start_symbol, self.productions[0][1], 0)
        initial_state = self._closure({initial_item})
        
        self.canonical_collection = [initial_state]
        worklist = [initial_state]
        
        while worklist:
            current_state = worklist.pop(0)
            current_state_idx = self.canonical_collection.index(current_state)
            
            all_symbols = self.terminals.union(self.non_terminals) - {'$'}
            for symbol in sorted(list(all_symbols)):
                next_state = self._goto(current_state, symbol)
                if next_state:
                    if next_state not in self.canonical_collection:
                        self.canonical_collection.append(next_state)
                        worklist.append(next_state)
                    
                    next_state_idx = self.canonical_collection.index(next_state)
                    self.goto_map[(current_state_idx, symbol)] = next_state_idx

    def _build_parsing_table(self):
        """TASK 2: Builds the SLR ACTION and GOTO tables."""
        for i in range(len(self.canonical_collection)):
            self.action_table[i] = {}
            self.goto_table[i] = {}

        for i, state in enumerate(self.canonical_collection):
            # GOTO entries (for non-terminals)
            for nt in self.non_terminals:
                if (i, nt) in self.goto_map:
                    self.goto_table[i][nt] = self.goto_map[(i, nt)]

            for item in state:
                head, body, dot_pos = item
                # SHIFT actions
                if dot_pos < len(body):
                    symbol = body[dot_pos]
                    if symbol in self.terminals and (i, symbol) in self.goto_map:
                        target_state = self.goto_map[(i, symbol)]
                        self.action_table[i][symbol] = ('shift', target_state)
                # REDUCE and ACCEPT actions
                else:
                    if head == self.augmented_start_symbol:
                        self.action_table[i]['$'] = ('accept', None)
                    else:
                        prod_idx = self.productions.index((head, body))
                        for term in self.follow_sets[head]:
                            self.action_table[i][term] = ('reduce', prod_idx)
    
    def print_productions(self):
        print("--- Grammar Productions ---")
        for i, (head, body) in enumerate(self.productions):
            print(f"{i}: {head} -> {' '.join(body)}")
        print("-" * 25)

    def print_canonical_collection(self):
        print("\n--- Canonical LR(0) Collection (Set of Items) ---")
        for i, state in enumerate(self.canonical_collection):
            print(f"I{i}:")
            # Sort for consistent output
            sorted_items = sorted(list(state), key=lambda x: (x[0], x[1]))
            for head, body, dot_pos in sorted_items:
                body_with_dot = list(body)
                body_with_dot.insert(dot_pos, '.')
                print(f"  {head} -> {' '.join(body_with_dot)}")
        print("-" * 50)
        
    def print_parsing_table(self):
        print("\n--- SLR Parsing Table (Action and GOTO) ---")
        action_terminals = sorted(list(self.terminals))
        goto_non_terminals = sorted(list(self.non_terminals - {self.augmented_start_symbol}))
        
        header = ['State'] + action_terminals + goto_non_terminals
        print(f"{header[0]:<6}" + "".join([f"| {h:<5}" for h in header[1:]]))
        print("-" * (6 + 7 * len(header[1:])))
        
        for i in range(len(self.canonical_collection)):
            row = [f"{i:<6}"]
            for term in action_terminals:
                action = self.action_table[i].get(term)
                if action:
                    if action[0] == 'shift': row.append(f"s{action[1]:<4}")
                    elif action[0] == 'reduce': row.append(f"r{action[1]:<4}")
                    elif action[0] == 'accept': row.append(f"{'acc':<5}")
                else: row.append("")
            for nt in goto_non_terminals:
                goto = self.goto_table[i].get(nt)
                row.append(f"{goto:<5}" if goto is not None else "")
            print(" | ".join(row))
        print("-" * 50)

    def parse(self, input_string):
        """Parses an input string using the generated table."""
        tokens = input_string.strip().split() + ['$']
        stack = [0]
        pointer = 0
        
        print("\n--- Parsing Trace ---")
        print(f"{'Stack':<30} | {'Input':<30} | {'Action'}")
        print("-" * 75)
        
        while True:
            state = stack[-1]
            token = tokens[pointer]
            
            stack_str = ' '.join(map(str, stack))
            input_str = ' '.join(tokens[pointer:])
            
            action = self.action_table[state].get(token)
            if not action:
                print(f"{stack_str:<30} | {input_str:<30} | Error: Reject")
                return "Reject"

            if action[0] == 'shift':
                action_str = f"Shift to {action[1]}"
                print(f"{stack_str:<30} | {input_str:<30} | {action_str}")
                stack.append(token)
                stack.append(action[1])
                pointer += 1
            elif action[0] == 'reduce':
                prod_idx = action[1]
                head, body = self.productions[prod_idx]
                action_str = f"Reduce by r{prod_idx} ({head} -> {' '.join(body)})"
                print(f"{stack_str:<30} | {input_str:<30} | {action_str}")
                
                for _ in range(2 * len(body)):
                    stack.pop()
                
                prev_state = stack[-1]
                goto_state = self.goto_table[prev_state][head]
                stack.append(head)
                stack.append(goto_state)
            elif action[0] == 'accept':
                print(f"{stack_str:<30} | {input_str:<30} | Accept")
                return "Accept"

# --- Main Execution ---
if __name__ == "__main__":
    # Grammar from the assignment slides, using '|' for OR on the same line
    assignment_grammar = """
E -> E + T | T
T -> T * F | F
F -> ( E ) | id
"""

    # Create the parser generator object
    parser = SLRParser(assignment_grammar)
    
    # Print numbered productions for reference
    parser.print_productions()

    # STEP 1: Generate and display the Set of Items
    parser.print_canonical_collection()
    
    # STEP 2: Generate and display the Action and GOTO table
    parser.print_parsing_table()
    
    # Demonstrate the parser with a sample string
    input_to_parse = "id + id * id"
    print(f"\n--- Demonstrating Parser on Input: '{input_to_parse}' ---")
    result = parser.parse(input_to_parse)
    print(f"\nFinal Result: {result}")