# UCS 802 COMPILER CONSTRUCTION LAB ASSIGNMENT 1
# Generate a Minimized DFA from a Regular Expression

# Epsilon symbol
EPSILON = 'ε'

class NFA:
    """Class to represent a Non-deterministic Finite Automaton."""
    def __init__(self, states, alphabet, transitions, start_state, final_states):
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions
        self.start_state = start_state
        self.final_states = final_states

    def __str__(self):
        return (
            f"NFA:\n"
            f"  States: {self.states}\n"
            f"  Alphabet: {self.alphabet}\n"
            f"  Start State: {self.start_state}\n"
            f"  Final States: {self.final_states}\n"
            f"  Transitions: {self.transitions}"
        )

class DFA:
    """Class to represent a Deterministic Finite Automaton."""
    def __init__(self, states, alphabet, transitions, start_state, final_states):
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions
        self.start_state = start_state
        self.final_states = final_states
    
    def __str__(self):
        return (
            f"DFA:\n"
            f"  States: {self.states}\n"
            f"  Alphabet: {self.alphabet}\n"
            f"  Start State: {self.start_state}\n"
            f"  Final States: {self.final_states}\n"
            f"  Transitions: {self.transitions}"
        )

# --- Step 1: Thompson's Construction (RE -> NFA) ---

def preprocess_regex(regex):
    """Inserts explicit concatenation operators '.' into the regex."""
    output = []
    for i in range(len(regex)):
        output.append(regex[i])
        if i + 1 < len(regex):
            # Insert '.' if current and next chars are operands or specific operators
            if (regex[i].isalnum() or regex[i] in ')*') and \
               (regex[i+1].isalnum() or regex[i+1] == '('):
                output.append('.')
    return "".join(output)

def infix_to_postfix(regex):
    """Converts an infix regular expression to postfix."""
    # Use '|' for union to avoid confusion with file paths
    regex = regex.replace('/', '|')
    preprocessed = preprocess_regex(regex)
    
    precedence = {'|': 1, '.': 2, '*': 3}
    postfix = []
    operator_stack = []

    for char in preprocessed:
        if char.isalnum():
            postfix.append(char)
        elif char == '(':
            operator_stack.append(char)
        elif char == ')':
            while operator_stack and operator_stack[-1] != '(':
                postfix.append(operator_stack.pop())
            operator_stack.pop() # Pop '('
        else: # Operator
            while (operator_stack and operator_stack[-1] != '(' and 
                   precedence.get(operator_stack[-1], 0) >= precedence.get(char, 0)):
                postfix.append(operator_stack.pop())
            operator_stack.append(char)

    while operator_stack:
        postfix.append(operator_stack.pop())
        
    return "".join(postfix)

def thompson_construction(postfix_regex):
    """Builds an NFA from a postfix regular expression."""
    nfa_stack = []
    state_counter = 0

    def new_state():
        nonlocal state_counter
        state = state_counter
        state_counter += 1
        return state

    for char in postfix_regex:
        if char.isalnum():
            # Base case: create NFA for a single character
            start = new_state()
            final = new_state()
            nfa = NFA(
                states={start, final},
                alphabet={char},
                transitions={start: {char: {final}}},
                start_state=start,
                final_states={final}
            )
            nfa_stack.append(nfa)
        elif char == '.':
            # Concatenation
            nfa2 = nfa_stack.pop()
            nfa1 = nfa_stack.pop()
            
            # Merge final state of nfa1 with start state of nfa2
            nfa1.transitions.update(nfa2.transitions)
            for s in nfa1.final_states:
                if s not in nfa1.transitions: nfa1.transitions[s] = {}
                nfa1.transitions[s][EPSILON] = {nfa2.start_state}
            
            nfa1.states.update(nfa2.states)
            nfa1.alphabet.update(nfa2.alphabet)
            nfa1.final_states = nfa2.final_states
            nfa_stack.append(nfa1)

        elif char == '|':
            # Union
            nfa2 = nfa_stack.pop()
            nfa1 = nfa_stack.pop()
            start = new_state()
            final = new_state()

            transitions = {start: {EPSILON: {nfa1.start_state, nfa2.start_state}}}
            transitions.update(nfa1.transitions)
            transitions.update(nfa2.transitions)
            for s in nfa1.final_states.union(nfa2.final_states):
                if s not in transitions: transitions[s] = {}
                transitions[s][EPSILON] = {final}

            nfa = NFA(
                states=nfa1.states.union(nfa2.states).union({start, final}),
                alphabet=nfa1.alphabet.union(nfa2.alphabet),
                transitions=transitions,
                start_state=start,
                final_states={final}
            )
            nfa_stack.append(nfa)

        elif char == '*':
            # Kleene Star
            nfa1 = nfa_stack.pop()
            start = new_state()
            final = new_state()
            
            transitions = {start: {EPSILON: {nfa1.start_state, final}}}
            transitions.update(nfa1.transitions)
            for s in nfa1.final_states:
                if s not in transitions: transitions[s] = {}
                transitions[s][EPSILON] = {nfa1.start_state, final}
                
            nfa = NFA(
                states=nfa1.states.union({start, final}),
                alphabet=nfa1.alphabet,
                transitions=transitions,
                start_state=start,
                final_states={final}
            )
            nfa_stack.append(nfa)

    return nfa_stack.pop()

# --- Step 2: Subset Construction (NFA -> DFA) ---

def epsilon_closure(nfa, states):
    """Computes the epsilon closure for a set of NFA states."""
    closure = set(states)
    stack = list(states)
    while stack:
        state = stack.pop()
        # Find states reachable by epsilon transitions
        epsilon_neighbors = nfa.transitions.get(state, {}).get(EPSILON, set())
        for neighbor in epsilon_neighbors:
            if neighbor not in closure:
                closure.add(neighbor)
                stack.append(neighbor)
    return frozenset(closure)

def move(nfa, states, char):
    """Computes the set of states reachable on a character from a set of states."""
    reachable_states = set()
    for state in states:
        reachable_states.update(nfa.transitions.get(state, {}).get(char, set()))
    return frozenset(reachable_states)

def subset_construction(nfa):
    """Converts an NFA to a DFA using the subset construction algorithm."""
    # The DFA states are sets of NFA states
    dfa_states = set()
    dfa_transitions = {}
    
    # The initial DFA state is the epsilon closure of the NFA's start state
    initial_dfa_state = epsilon_closure(nfa, {nfa.start_state})
    dfa_states.add(initial_dfa_state)
    
    worklist = [initial_dfa_state]
    
    while worklist:
        current_dfa_state = worklist.pop(0)
        dfa_transitions[current_dfa_state] = {}
        
        for char in nfa.alphabet:
            next_nfa_states = move(nfa, current_dfa_state, char)
            next_dfa_state = epsilon_closure(nfa, next_nfa_states)
            
            if not next_dfa_state: # Trap state if empty
                continue

            if next_dfa_state not in dfa_states:
                dfa_states.add(next_dfa_state)
                worklist.append(next_dfa_state)
            
            dfa_transitions[current_dfa_state][char] = next_dfa_state

    # Map frozensets to integer state names for clarity
    state_map = {state: i for i, state in enumerate(dfa_states)}
    
    # Determine final states of the DFA
    dfa_final_states = {
        state_map[s] for s in dfa_states if not nfa.final_states.isdisjoint(s)
    }

    # Build the final DFA object
    return DFA(
        states=set(state_map.values()),
        alphabet=nfa.alphabet,
        transitions={state_map[s]: {c: state_map[t] for c, t in trans.items()} for s, trans in dfa_transitions.items()},
        start_state=state_map[initial_dfa_state],
        final_states=dfa_final_states
    )

# --- Step 3: DFA Minimization ---

def minimize_dfa(dfa):
    """Minimizes a DFA using the partition refinement algorithm."""
    # Initial partition: final states and non-final states
    final_states = frozenset(dfa.final_states)
    non_final_states = frozenset(dfa.states - dfa.final_states)
    partitions = {final_states, non_final_states}
    worklist = {final_states, non_final_states}

    # Filter out empty sets if all states are final or non-final
    partitions = {p for p in partitions if p}
    worklist = {w for w in worklist if w}

    while worklist:
        A = worklist.pop()
        for char in dfa.alphabet:
            # X is the set of states that transition into A on character 'char'
            X = frozenset({s for s in dfa.states if dfa.transitions.get(s, {}).get(char) in A})

            new_partitions = set()
            for Y in partitions:
                intersection = Y.intersection(X)
                difference = Y.difference(X)
                if intersection and difference:
                    # Split Y
                    new_partitions.add(intersection)
                    new_partitions.add(difference)
                    if Y in worklist:
                        worklist.remove(Y)
                        worklist.add(intersection)
                        worklist.add(difference)
                    else:
                        if len(intersection) <= len(difference):
                            worklist.add(intersection)
                        else:
                            worklist.add(difference)
                else:
                    # No split, keep Y as is
                    new_partitions.add(Y)
            partitions = new_partitions
    
    # Create the minimized DFA from the final partitions
    min_state_map = {frozenset(p): i for i, p in enumerate(partitions)}
    min_states = set(min_state_map.values())
    min_alphabet = dfa.alphabet
    
    min_start_state = None
    for p, i in min_state_map.items():
        if dfa.start_state in p:
            min_start_state = i
            break
            
    min_final_states = {min_state_map[p] for p in partitions if not p.isdisjoint(dfa.final_states)}
    
    min_transitions = {}
    for p, i in min_state_map.items():
        # Pick a representative state from the partition
        representative = next(iter(p))
        min_transitions[i] = {}
        for char in min_alphabet:
            target_state = dfa.transitions.get(representative, {}).get(char)
            if target_state is not None:
                for target_p, target_i in min_state_map.items():
                    if target_state in target_p:
                        min_transitions[i][char] = target_i
                        break
                        
    return DFA(min_states, min_alphabet, min_transitions, min_start_state, min_final_states)

# --- DFA Simulation ---

def simulate_dfa(dfa, input_string):
    """Simulates a DFA on an input string."""
    current_state = dfa.start_state
    for char in input_string:
        if char not in dfa.alphabet:
            return False # Character not in alphabet
        
        current_state = dfa.transitions.get(current_state, {}).get(char)
        if current_state is None:
            return False # No transition defined (implicit trap state)
    
    return current_state in dfa.final_states

# --- Main Execution ---

if __name__ == "__main__":
    # The regular expression from the assignment
    regex = "(a/b)*abb"
    print(f"Regular Expression: {regex}\n")

    # --- Step 1 ---
    print("--- Step 1: RE to NFA (Thompson's Construction) ---")
    postfix = infix_to_postfix(regex)
    print(f"Postfix Expression: {postfix}")
    nfa = thompson_construction(postfix)
    print(nfa)
    print("-" * 50)

    # --- Step 2 ---
    print("--- Step 2: NFA to DFA (Subset Construction) ---")
    unminimized_dfa = subset_construction(nfa)
    print("Generated (Unminimized) DFA:")
    print(unminimized_dfa)
    print("-" * 50)
    
    # --- Step 3 ---
    print("--- Step 3: DFA Minimization ---")
    minimized_dfa = minimize_dfa(unminimized_dfa)
    print("Minimized DFA:")
    print(minimized_dfa)
    print("-" * 50)
    
    # --- Final Output and Testing ---
    print("--- Testing the Minimized DFA ---")
    test_strings = ["abb", "aabb", "babb", "ab", "a", "banana", "", "abbabb"]
    
    for s in test_strings:
        result = simulate_dfa(minimized_dfa, s)
        output = "Accept" if result else "Not Accepted"
        print(f"Input: '{s}' -> Output: '{output}'")