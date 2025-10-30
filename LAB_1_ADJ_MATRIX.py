# Assignment 1 - RE to Minimized DFA with Adjacency Matrix Output (CORRECTED)
import copy

# Epsilon symbol
EPSILON = 'ε'

class NFA:
    """Class to represent a Non-deterministic Finite Automaton."""
    def __init__(self, states, alphabet, transitions, start_state, final_states):
        # CORRECTED: Store states as a set for union operations
        self.states = set(states)
        self.alphabet = alphabet
        self.transitions = transitions
        self.start_state = start_state
        self.final_states = final_states

    def print_adjacency_matrix(self):
        """Prints the NFA's transition table as an adjacency matrix."""
        print("\n--- NFA Adjacency Matrix ---")
        # CORRECTED: Create a temporary sorted list for consistent printing
        sorted_states = sorted(list(self.states))
        num_states = len(sorted_states)
        matrix = [['-' for _ in range(num_states)] for _ in range(num_states)]

        for from_state, trans_dict in self.transitions.items():
            for symbol, to_states_set in trans_dict.items():
                for to_state in to_states_set:
                    # Use the sorted list to find the index
                    from_idx = sorted_states.index(from_state)
                    to_idx = sorted_states.index(to_state)
                    
                    if matrix[from_idx][to_idx] == '-':
                        matrix[from_idx][to_idx] = symbol
                    else:
                        matrix[from_idx][to_idx] += f",{symbol}"

        # Print header using the sorted list
        header = "      " + " ".join([f"q{s:<2}" for s in sorted_states])
        print(header)
        print("    " + "-" * (len(header) - 4))
        # Print matrix rows
        for i, row in enumerate(matrix):
            row_str = f"q{sorted_states[i]:<2} | " + " ".join([f"{cell:<3}" for cell in row])
            print(row_str)
        print("-" * 50)


class DFA:
    """Class to represent a Deterministic Finite Automaton."""
    def __init__(self, states, alphabet, transitions, start_state, final_states):
        # CORRECTED: Store states as a set
        self.states = set(states)
        self.alphabet = alphabet
        self.transitions = transitions
        self.start_state = start_state
        self.final_states = final_states
    
    def print_adjacency_matrix(self):
        """Prints the DFA's transition table as an adjacency matrix."""
        print(f"\n--- DFA Adjacency Matrix ({'Minimized' if 'minimized' in self.__class__.__name__.lower() else 'Unminimized'}) ---")
        # CORRECTED: Create a temporary sorted list for printing
        sorted_states = sorted(list(self.states))
        num_states = len(sorted_states)
        matrix = [['-' for _ in range(num_states)] for _ in range(num_states)]

        for from_state, trans_dict in self.transitions.items():
            for symbol, to_state in trans_dict.items():
                from_idx = sorted_states.index(from_state)
                to_idx = sorted_states.index(to_state)
                
                if matrix[from_idx][to_idx] == '-':
                    matrix[from_idx][to_idx] = symbol
                else:
                    matrix[from_idx][to_idx] += f",{symbol}"
        
        # Print header
        header = "      " + " ".join([f"S{s:<2}" for s in sorted_states])
        print(header)
        print("    " + "-" * (len(header) - 4))
        # Print matrix rows
        for i, row in enumerate(matrix):
            row_str = f"S{sorted_states[i]:<2} | " + " ".join([f"{cell:<3}" for cell in row])
            print(row_str)
        print("-" * 50)


# (The rest of the functions are unchanged and correct)

def preprocess_regex(regex):
    output = []
    for i in range(len(regex)):
        output.append(regex[i])
        if i + 1 < len(regex):
            if (regex[i].isalnum() or regex[i] in ')*') and \
               (regex[i+1].isalnum() or regex[i+1] == '('):
                output.append('.')
    return "".join(output)

def infix_to_postfix(regex):
    regex = regex.replace('/', '|')
    preprocessed = preprocess_regex(regex)
    precedence = {'|': 1, '.': 2, '*': 3}
    postfix, operator_stack = [], []
    for char in preprocessed:
        if char.isalnum(): postfix.append(char)
        elif char == '(': operator_stack.append(char)
        elif char == ')':
            while operator_stack and operator_stack[-1] != '(':
                postfix.append(operator_stack.pop())
            operator_stack.pop()
        else:
            while (operator_stack and operator_stack[-1] != '(' and 
                   precedence.get(operator_stack[-1], 0) >= precedence.get(char, 0)):
                postfix.append(operator_stack.pop())
            operator_stack.append(char)
    while operator_stack: postfix.append(operator_stack.pop())
    return "".join(postfix)

def thompson_construction(postfix_regex):
    nfa_stack = []
    state_counter = 0
    def new_state():
        nonlocal state_counter
        state = state_counter
        state_counter += 1
        return state
    for char in postfix_regex:
        if char.isalnum():
            start, final = new_state(), new_state()
            nfa_stack.append(NFA({start, final}, {char}, {start: {char: {final}}}, start, {final}))
        elif char == '.':
            nfa2, nfa1 = nfa_stack.pop(), nfa_stack.pop()
            nfa1.transitions.update(nfa2.transitions)
            for s in nfa1.final_states:
                nfa1.transitions.setdefault(s, {})[EPSILON] = {nfa2.start_state}
            # Now this union operation will work correctly
            nfa1.states.update(nfa2.states); nfa1.alphabet.update(nfa2.alphabet); nfa1.final_states = nfa2.final_states
            nfa_stack.append(nfa1)
        elif char == '|':
            nfa2, nfa1 = nfa_stack.pop(), nfa_stack.pop()
            start, final = new_state(), new_state()
            transitions = {start: {EPSILON: {nfa1.start_state, nfa2.start_state}}}
            transitions.update(nfa1.transitions); transitions.update(nfa2.transitions)
            for s in nfa1.final_states.union(nfa2.final_states):
                transitions.setdefault(s, {})[EPSILON] = {final}
            nfa_stack.append(NFA(nfa1.states.union(nfa2.states).union({start, final}),
                                 nfa1.alphabet.union(nfa2.alphabet), transitions, start, {final}))
        elif char == '*':
            nfa1 = nfa_stack.pop()
            start, final = new_state(), new_state()
            transitions = {start: {EPSILON: {nfa1.start_state, final}}}
            transitions.update(nfa1.transitions)
            for s in nfa1.final_states:
                transitions.setdefault(s, {})[EPSILON] = {nfa1.start_state, final}
            nfa_stack.append(NFA(nfa1.states.union({start, final}), nfa1.alphabet,
                                 transitions, start, {final}))
    return nfa_stack.pop()

def epsilon_closure(nfa, states):
    closure, stack = set(states), list(states)
    while stack:
        state = stack.pop()
        for neighbor in nfa.transitions.get(state, {}).get(EPSILON, set()):
            if neighbor not in closure:
                closure.add(neighbor); stack.append(neighbor)
    return frozenset(closure)

def move(nfa, states, char):
    reachable = set()
    for state in states:
        reachable.update(nfa.transitions.get(state, {}).get(char, set()))
    return frozenset(reachable)

def subset_construction(nfa):
    dfa_states, dfa_transitions = set(), {}
    initial_dfa_state = epsilon_closure(nfa, {nfa.start_state})
    dfa_states.add(initial_dfa_state)
    worklist = [initial_dfa_state]
    while worklist:
        current_dfa_state = worklist.pop(0)
        dfa_transitions[current_dfa_state] = {}
        for char in nfa.alphabet:
            next_nfa_states = move(nfa, current_dfa_state, char)
            next_dfa_state = epsilon_closure(nfa, next_nfa_states)
            if next_dfa_state:
                if next_dfa_state not in dfa_states:
                    dfa_states.add(next_dfa_state); worklist.append(next_dfa_state)
                dfa_transitions[current_dfa_state][char] = next_dfa_state
    state_map = {state: i for i, state in enumerate(dfa_states)}
    final_states = {state_map[s] for s in dfa_states if nfa.final_states and not nfa.final_states.isdisjoint(s)}
    return DFA(set(state_map.values()), nfa.alphabet,
               {state_map[s]: {c: state_map[t] for c, t in trans.items()} for s, trans in dfa_transitions.items()},
               state_map[initial_dfa_state], final_states)

def minimize_dfa(dfa):
    final, non_final = frozenset(dfa.final_states), frozenset(dfa.states - dfa.final_states)
    partitions = {p for p in {final, non_final} if p}
    worklist = list(partitions)
    while worklist:
        A = worklist.pop(0)
        for char in dfa.alphabet:
            X = frozenset({s for s in dfa.states if dfa.transitions.get(s, {}).get(char) in A})
            new_partitions, temp_partitions = set(), list(partitions)
            for Y in temp_partitions:
                intersection, difference = Y & X, Y - X
                if intersection and difference:
                    new_partitions.add(intersection); new_partitions.add(difference)
                    if Y in worklist: worklist.remove(Y)
                    worklist.append(intersection); worklist.append(difference)
                else: new_partitions.add(Y)
            partitions = new_partitions
    min_state_map = {frozenset(p): i for i, p in enumerate(partitions)}
    min_start = next(i for p, i in min_state_map.items() if dfa.start_state in p)
    min_finals = {min_state_map[p] for p in partitions if dfa.final_states and not p.isdisjoint(dfa.final_states)}
    min_trans = {}
    for p, i in min_state_map.items():
        rep, min_trans[i] = next(iter(p)), {}
        for char, target in dfa.transitions.get(rep, {}).items():
            min_trans[i][char] = min_state_map[next(part for part in partitions if target in part)]
    min_dfa = DFA(set(min_state_map.values()), dfa.alphabet, min_trans, min_start, min_finals)
    min_dfa.__class__.__name__ = 'MinimizedDFA' # For custom printing
    return min_dfa

def simulate_dfa_with_trace(dfa, input_string):
    """Simulates a DFA and shows the matching steps."""
    print(f"\n--- String Matching Steps for '{input_string}' ---")
    current_state = dfa.start_state
    
    if not input_string:
        is_accepted = current_state in dfa.final_states
        print(f"Input is empty. Start state S{current_state} is {'final' if is_accepted else 'not final'}.")
        return "Accept" if is_accepted else "Not Accepted"

    for i, char in enumerate(input_string):
        if char not in dfa.alphabet:
            print(f"Step {i+1}: Character '{char}' not in alphabet. Rejecting.")
            return "Not Accepted"
        
        next_state = dfa.transitions.get(current_state, {}).get(char)
        
        if next_state is None:
            print(f"Step {i+1}: From state S{current_state}, no transition on '{char}'. Rejecting.")
            return "Not Accepted"
            
        print(f"Step {i+1}: Start in state S{current_state}, read '{char}', move to state S{next_state}")
        current_state = next_state
    
    is_accepted = current_state in dfa.final_states
    print(f"End of string. Final state is S{current_state}, which is {'an accepting' if is_accepted else 'not an accepting'} state.")
    return "Accept" if is_accepted else "Not Accepted"


# --- Main Execution ---
if __name__ == "__main__":
    regex_input = "(a/b)*abb"
    print(f"Processing Regular Expression: {regex_input}\n")

    # --- Step 1: RE to NFA ---
    postfix = infix_to_postfix(regex_input)
    nfa = thompson_construction(postfix)
    nfa.print_adjacency_matrix()

    # --- Step 2: NFA to DFA ---
    unminimized_dfa = subset_construction(nfa)
    unminimized_dfa.print_adjacency_matrix()
    
    # --- Step 3: DFA Minimization ---
    minimized_dfa = minimize_dfa(unminimized_dfa)
    minimized_dfa.print_adjacency_matrix()
    
    # --- Final Test Cases ---
    test_strings = ["abb", "aabb", "ab", "b"]
    for s in test_strings:
        final_result = simulate_dfa_with_trace(minimized_dfa, s)
        print(f"Final Result for '{s}': {final_result}")
        print("-" * 50)