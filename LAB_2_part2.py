# UCS 802 Compiler Construction: Lab Assignment II
# A Generic SLR Parser Generator

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
        lines = [line for line in grammar_str.strip().split('\n') if line.strip()]
        if not lines:
            raise ValueError("Grammar is empty!")
        self.start_symbol = lines[0].split('->')[0].strip()
        self.augmented_start_symbol = self.start_symbol + "'"
        
        self.grammar[self.augmented_start_symbol] = [tuple([self.start_symbol])]
        self.productions.append((self.augmented_start_symbol, tuple([self.start_symbol])))

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
                    if not symbol[0].isupper():
                        self.terminals.add(symbol)
        
        self.non_terminals.add(self.augmented_start_symbol)
        self.terminals.add('$')

    def _compute_first_and_follow(self):
        """Computes FIRST and FOLLOW sets for all non-terminals."""
        for nt in self.non_terminals:
            self.first_sets[nt] = set()
        
        while True:
            updated = False
            for head, bodies in self.grammar.items():
                for body in bodies:
                    for symbol in body:
                        original_size = len(self.first_sets[head])
                        if symbol in self.terminals:
                            self.first_sets[head].add(symbol)
                            break
                        else:
                            self.first_sets[head].update(self.first_sets[symbol])
                            if 'epsilon' not in self.first_sets[symbol]: break
                    if len(self.first_sets[head]) > original_size: updated = True
            if not updated: break

        for nt in self.non_terminals:
            self.follow_sets[nt] = set()
        self.follow_sets[self.start_symbol].add('$')

        while True:
            updated = False
            for head, bodies in self.grammar.items():
                for body in bodies:
                    for i, symbol in enumerate(body):
                        if symbol in self.non_terminals:
                            original_size = len(self.follow_sets[symbol])
                            trailer = body[i+1:]
                            if trailer:
                                first_of_trailer = set()
                                for t_symbol in trailer:
                                    if t_symbol in self.terminals:
                                        first_of_trailer.add(t_symbol); break
                                    else:
                                        first_of_trailer.update(self.first_sets[t_symbol])
                                        if 'epsilon' not in self.first_sets[t_symbol]: break
                                self.follow_sets[symbol].update(first_of_trailer)
                            else:
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
                symbol = body[dot_pos]
                if symbol in self.non_terminals:
                    for prod_body in self.grammar.get(symbol, []):
                        new_item = (symbol, prod_body, 0)
                        if new_item not in closure_set:
                            closure_set.add(new_item); worklist.append(new_item)
        return frozenset(closure_set)

    def _goto(self, items, symbol):
        """Computes the GOTO set."""
        new_items = set()
        for head, body, dot_pos in items:
            if dot_pos < len(body) and body[dot_pos] == symbol:
                new_items.add((head, body, dot_pos + 1))
        return self._closure(new_items)

    def _build_canonical_collection(self):
        """Builds the canonical collection of LR(0) items."""
        initial_item = (self.augmented_start_symbol, self.productions[0][1], 0)
        initial_state = self._closure({initial_item})
        
        self.canonical_collection = [initial_state]
        worklist = [initial_state]
        
        while worklist:
            current_state = worklist.pop(0)
            current_idx = self.canonical_collection.index(current_state)
            all_symbols = self.terminals.union(self.non_terminals) - {'$'}
            for symbol in sorted(list(all_symbols)):
                next_state = self._goto(current_state, symbol)
                if next_state:
                    if next_state not in self.canonical_collection:
                        self.canonical_collection.append(next_state); worklist.append(next_state)
                    self.goto_map[(current_idx, symbol)] = self.canonical_collection.index(next_state)

    def _build_parsing_table(self):
        """Builds the SLR ACTION and GOTO tables."""
        for i in range(len(self.canonical_collection)):
            self.action_table[i] = {}; self.goto_table[i] = {}
        
        for i, state in enumerate(self.canonical_collection):
            for nt in self.non_terminals:
                if (i, nt) in self.goto_map: self.goto_table[i][nt] = self.goto_map[(i, nt)]

            for head, body, dot_pos in state:
                if dot_pos < len(body):
                    symbol = body[dot_pos]
                    if symbol in self.terminals and (i, symbol) in self.goto_map:
                        target = self.goto_map[(i, symbol)]
                        if symbol in self.action_table[i] and self.action_table[i][symbol] != ('shift', target):
                            print(f"Conflict at state {i} on '{symbol}': {self.action_table[i][symbol]} vs ('shift', {target})")
                        self.action_table[i][symbol] = ('shift', target)
                else:
                    if head == self.augmented_start_symbol:
                        self.action_table[i]['$'] = ('accept', None)
                    else:
                        prod_idx = self.productions.index((head, body))
                        for term in self.follow_sets[head]:
                            if term in self.action_table[i] and self.action_table[i][term] != ('reduce', prod_idx):
                                print(f"Conflict at state {i} on '{term}': {self.action_table[i][term]} vs ('reduce', {prod_idx})")
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
            for head, body, dot_pos in sorted(list(state)):
                body_with_dot = list(body); body_with_dot.insert(dot_pos, '.')
                print(f"  {head} -> {' '.join(body_with_dot)}")
        print("-" * 50)
        
    def print_parsing_table(self):
        print("\n--- SLR Parsing Table (Action and GOTO) ---")
        action_terms = sorted(list(self.terminals))
        goto_non_terms = sorted(list(self.non_terminals - {self.augmented_start_symbol}))
        header = ['State'] + action_terms + goto_non_terms
        print(f"{header[0]:<6}" + "".join([f"| {h:<5}" for h in header[1:]]))
        print("-" * (6 + 7 * len(header[1:])))
        
        for i in range(len(self.canonical_collection)):
            row = [f"{i:<6}"]
            for term in action_terms:
                action = self.action_table[i].get(term)
                if action:
                    if action[0]=='shift': row.append(f"s{action[1]:<4}")
                    elif action[0]=='reduce': row.append(f"r{action[1]:<4}")
                    elif action[0]=='accept': row.append("acc")
                else: row.append("")
            for nt in goto_non_terms:
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
            state, token = stack[-1], tokens[pointer]
            stack_str, input_str = ' '.join(map(str, stack)), ' '.join(tokens[pointer:])
            
            action = self.action_table[state].get(token)
            if not action:
                print(f"{stack_str:<30} | {input_str:<30} | Error: Reject"); return "Reject"

            if action[0] == 'shift':
                print(f"{stack_str:<30} | {input_str:<30} | Shift to {action[1]}")
                stack.extend([token, action[1]]); pointer += 1
            elif action[0] == 'reduce':
                prod_idx = action[1]
                head, body = self.productions[prod_idx]
                print(f"{stack_str:<30} | {input_str:<30} | Reduce by r{prod_idx} ({head} -> {' '.join(body)})")
                stack = stack[:-2 * len(body)]
                prev_state = stack[-1]
                goto_state = self.goto_table[prev_state][head]
                stack.extend([head, goto_state])
            elif action[0] == 'accept':
                print(f"{stack_str:<30} | {input_str:<30} | Accept"); return "Accept"

# --- Generic Main Execution ---
if __name__ == "__main__":
    print("--- Generic SLR Parser Generator ---")
    
    print("\nEnter your grammar (one production per line, use '|' for alternatives).")
    print("Press Enter on an empty line when you are done.")
    grammar_lines = []
    while True:
        line = input()
        if not line:
            break
        grammar_lines.append(line)
    grammar_input = "\n".join(grammar_lines)
    
    string_to_parse = input("\nEnter the string to parse (tokens separated by spaces): ")

    try:
        print("\n--- Initializing Parser ---")
        parser = SLRParser(grammar_input)
        
        parser.print_productions()
        parser.print_canonical_collection()
        parser.print_parsing_table()
        
        print(f"\n--- Parsing Input: '{string_to_parse}' ---")
        result = parser.parse(string_to_parse)
        print(f"\nFinal Result: {result}")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please check the grammar format is correct and does not have conflicts.")