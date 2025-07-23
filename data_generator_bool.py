import random
import pickle

class BoolLogicTokenizer:
    def __init__(self):
        self.tokens = {
            ' ': 0,  # padding token
            'E': 1,  # end of expression
            'T': 2,
            'F': 3,
            '¬': 4,
            '∧': 5,
            '∨': 6,
            '→': 7,
            '↔': 8,
            '(': 9,
            ')': 10,
            '0': 11,  # Placeholder thinking tokens
            '1': 12,
            '2': 13,
            '3': 14,
            '4': 15,
        }
        self.reverse_tokens = {v: k for k, v in self.tokens.items()}

    def tokenize(self, expression):
        return [self.tokens[char] for char in expression if char in self.tokens]

    def detokenize(self, token_list):
        return ''.join(self.reverse_tokens[token] for token in token_list)


class BoolLogic:

    NOT = '¬'
    AND = '∧'
    OR = '∨'
    IMPLIES = '→'
    IFF = '↔'

    D = {
        'T': '¬F',
        'F': '¬T',
    }

    T_table = ['T∧T', 'T∨T', 'T∨F', 'F∨T']
    F_table = ['F∧T', 'T∧F', 'F∧F', 'F∨F']

    expanded_T_table = list()
    for item in T_table:
        new_item_1 = D[item[0]] + item[1:]
        new_item_2 = item[:2] + D[item[2]]
        new_item_3 = D[item[0]] + item[1] + D[item[2]]

        expanded_T_table.extend([item, new_item_1, new_item_2, new_item_3])

    expanded_F_table = list()
    for item in F_table:
        new_item_1 = D[item[0]] + item[1:]
        new_item_2 = item[:2] + D[item[2]]
        new_item_3 = D[item[0]] + item[1] + D[item[2]]

        expanded_F_table.extend([item, new_item_1, new_item_2, new_item_3])

    D_table = {
        'T': expanded_T_table,
        'F': expanded_F_table}

    @staticmethod
    def generate_expression(depth=3, verbose=1):

        origin = random.choice(['T', 'F'])
        current = origin
        expression = origin
        # current = random.choice(BoolLogic.D_table[origin])
        # expression = f"{current} {BoolLogic.IMPLIES} {origin}"
        
        for d in range(0, depth):
            
            recorded = current
            while recorded == current:
                new_current = []
                for t in list(current):
                    if t in ['T', 'F']:
                        if random.random() < 0.75:
                            t = random.choice(BoolLogic.D_table[t])
                            t = f"({t})" if d > 0 else t

                    new_current.append(t)
                current = ''.join(new_current)

            if (d + 1) % verbose == 0 or d == depth - 1:
                expression = f"{current}{BoolLogic.IMPLIES}{expression}"

        return expression


    @staticmethod
    def generate_expressions(num_samples=10, depth=3, verbose=1):

        expressions = set()
        while len(expressions) < num_samples:
            expressions.add(BoolLogic.generate_expression(depth, verbose))

        return list(expressions)
    
    @staticmethod
    def generate_dataset_and_save(num_samples=1000000, depth=5, verbose=1, filename='bool_logic_dataset.pkl'):
        expressions = BoolLogic.generate_expressions(num_samples, depth, verbose)
        tokenized_expressions = []
        tokenizer = BoolLogicTokenizer()
        pos_implies = []

        for e in expressions:
            tokenized_experssion = tokenizer.tokenize(e)
            tokenized_experssion.append(tokenizer.tokens['E'])  # Append end token

            first_implies = tokenized_experssion.index(tokenizer.tokens[BoolLogic.IMPLIES])
            pos_implies.append(first_implies)

        dataset = dict()
        dataset['expressions'] = expressions
        dataset['tokenized_expressions'] = tokenized_expressions
        dataset['pos_implies'] = pos_implies

        with open(filename, 'wb') as f:
            pickle.dump(tokenized_expressions, f)

    @staticmethod
    def load_dataset(filename='bool_logic_dataset.pkl'):
        with open(filename, 'rb') as f:
            tokenized_expressions = pickle.load(f)
        return tokenized_expressions

if __name__ == "__main__":
    # Example usage
    expressions = BoolLogic.generate_expressions(num_samples=1000, depth=5, verbose=1)
    print(max(len(expr) for expr in expressions))
    for expr in expressions:
        print(expr)

