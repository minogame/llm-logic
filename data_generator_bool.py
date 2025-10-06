import random
import pickle
import itertools
import numpy as np

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
        self.vocab_size = len(self.tokens)

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

        # expressions = set()
        # while len(expressions) < num_samples:
        #     expressions.add(BoolLogic.generate_expression(depth, verbose))

        # return list(expressions)

        expressions = []
        while len(expressions) < num_samples:
            expr = BoolLogic.generate_expression(depth, verbose)
            expressions.append(expr)

        return expressions
    
    @staticmethod
    def generate_dataset_and_save(num_samples=1000000, depth=5, verbose=1, filename='bool_logic_dataset.pkl'):
        print(f"Generating {num_samples} expressions with depth {depth} and verbosity {verbose}...")
        expressions = BoolLogic.generate_expressions(num_samples, depth, verbose)
        tokenized_expressions = []
        tokenizer = BoolLogicTokenizer()
        pos_implies = []

        print(f"Tokenizing expressions...")
        for e in expressions:
            tokenized_expression = tokenizer.tokenize(e)
            tokenized_expression.append(tokenizer.tokens['E'])  # Append end token

            tokenized_expressions.append(tokenized_expression)

            first_implies = tokenized_expression.index(tokenizer.tokens[BoolLogic.IMPLIES])
            pos_implies.append(first_implies)

        dataset = dict()
        dataset['expressions'] = expressions
        dataset['tokenized_expressions'] = tokenized_expressions
        dataset['pos_implies'] = pos_implies

        with open(filename, 'wb') as f:
            pickle.dump(dataset, f)

    @staticmethod
    def generate_mixed_dataset_and_save(num_samples=1000000, depth=[1,2,3,4,5,6], verbose=1, filename='bool_logic_dataset.pkl'):
        
        tokenizer = BoolLogicTokenizer()
        tokenized_expressions = []
        pos_implies = []

        expressions = []
        for d, v in itertools.product(depth, verbose):

            print(f"Generating {num_samples} expressions with depth {d} and verbosity {v}...")
            if d == 1:
                expressions.extend(BoolLogic.generate_expressions(16, d, v))
            elif d == 2:
                expressions.extend(BoolLogic.generate_expressions(256, d, v))
            elif d == 3:
                expressions.extend(BoolLogic.generate_expressions(num_samples, d, v))
            else:
                expressions.extend(BoolLogic.generate_expressions(num_samples, d, v))

        random.shuffle(expressions)

        print(f"Tokenizing expressions...")
        for e in expressions:
            tokenized_expression = tokenizer.tokenize(e)
            tokenized_expression.append(tokenizer.tokens['E'])  # Append end token

            tokenized_expressions.append(tokenized_expression)

            first_implies = tokenized_expression.index(tokenizer.tokens[BoolLogic.IMPLIES])
            pos_implies.append(first_implies)

        dataset = dict()
        dataset['expressions'] = expressions
        dataset['tokenized_expressions'] = tokenized_expressions
        dataset['pos_implies'] = pos_implies

        with open(filename, 'wb') as f:
            pickle.dump(dataset, f)

    @staticmethod
    def load_dataset(filename='bool_logic_dataset.pkl'):
        with open(filename, 'rb') as f:
            tokenized_expressions = pickle.load(f)
        return tokenized_expressions

    @staticmethod
    def generate_init_expression(expr, tokenizer, max_len=1536):
        _expr = expr.split(BoolLogic.IMPLIES)[0]
        ans = expr.split(BoolLogic.IMPLIES)[-1]

        all_tokens = list(range(2, 16))
        gen_length = max_len - len(_expr) - 4

        rand_len = min(gen_length, np.random.randint(len(_expr)*0.5, len(_expr)*2))
        expr_ = tokenizer.detokenize([random.choice(all_tokens) for _ in range(rand_len)])
        expr = _expr + BoolLogic.IMPLIES + expr_ + BoolLogic.IMPLIES + ans

        return expr, 1

    @staticmethod
    def evaluate_expression_1(expr, expr_):

        score = 0.0
        label = expr[-2:] # ->T or ->F
        label_ = expr_[-2:] # ->T or ->F

        if label == label_:
            score += 1.5
        else:
            if label == '→T' or label == '→F':
                score -= 0.5
            else:
                score -= 1.0
            
        return score

    @staticmethod
    def evaluate_expression_2(expr, expr_):

        score = 0.0
        label = expr[-2:] # ->T or ->F
        label_ = expr_[-2:] # ->T or ->F

        len_prompt = len(expr.split(BoolLogic.IMPLIES)[0])
        len_sample = len(expr) - len_prompt
        len_sample_ = len(expr_) - len_prompt

        if len_sample < len_sample_ // 2:
            score -= 0.5
        if len_sample < len_sample_ // 4:
            score -= 1.0
        if len_sample > len_sample_ * 2:
            score -= 0.5

        if label == label_:
            score += 1.5
        else:
            if label == '→T' or label == '→F':
                score -= 0.5
            else:
                score -= 0.5
            
        return score

    @staticmethod
    def evaluate_expression(expr, expr_, version=2):
        if version == 1:
            score = BoolLogic.evaluate_expression_1(expr, expr_)
        elif version == 2:
            score = BoolLogic.evaluate_expression_2(expr, expr_)
        else:
            raise NotImplementedError    
        
        return score
    
    @staticmethod
    def compute_statistics(score):
        statistics = (
            sum(1 for s in score if s == 1),      # num_correct
            sum(1 for s in score if s == -1),     # num_incorrect
            sum(1 for s in score if s == -2)      # num_invalid
        )
        return statistics

    @staticmethod
    def generate_init_expression_old(expr, tokenizer, max_len=1536):
        _expr = expr.split(BoolLogic.IMPLIES)[0]

        all_tokens = list(range(1, 16))
        gen_length = max_len - len(_expr)

        success = False
        attempts = 0

        while not success:
            attempts += 1
            expr_ = tokenizer.detokenize([random.choice(all_tokens) for _ in range(gen_length)])
            if '→TE' in expr_:
                expr_ = expr_.split('→TE')[0]
                expr  = _expr + BoolLogic.IMPLIES + expr_ + '→TE'
                success = True
            elif '→FE' in expr_:
                expr_ = expr_.split('→FE')[0]
                expr  = _expr + BoolLogic.IMPLIES + expr_ + '→FE'
                success = True
            else:
                pass
                
        return expr, attempts


    @staticmethod
    def generate_init_expressions_and_save(num_samples=1000000, depth=[3,4,5], filename='bool_logic_dataset.pkl'):
        tokenizer = BoolLogicTokenizer()
        tokenized_expressions = []
        pos_implies = []

        expressions = []

        n_t = 0

        while len(expressions) < num_samples:
            d = random.choice(depth)
            expr = BoolLogic.generate_expression(d, verbose = 999)
            expr, n = BoolLogic.generate_init_expression(expr, tokenizer)
            expressions.append(expr),
            n_t += n

        print(f"Tokenizing expressions...")
        for e in expressions:
            tokenized_expression = tokenizer.tokenize(e)
            tokenized_expression.append(tokenizer.tokens['E'])  # Append end token

            tokenized_expressions.append(tokenized_expression)

            first_implies = tokenized_expression.index(tokenizer.tokens[BoolLogic.IMPLIES])
            pos_implies.append(first_implies)

        dataset = dict()
        dataset['expressions'] = expressions
        dataset['tokenized_expressions'] = tokenized_expressions
        dataset['pos_implies'] = pos_implies

        with open(filename, 'wb') as f:
            pickle.dump(dataset, f)

if __name__ == "__main__":
    # # Example usage
    # expressions = BoolLogic.generate_expressions(num_samples=1000, depth=5, verbose=1)
    # print(max(len(expr) for expr in expressions))
    # for expr in expressions:
    #     print(expr)

    # BoolLogic.generate_mixed_dataset_and_save(num_samples=2000000, depth=[3,4,5,6], verbose=[1,2,3], filename='bool_logic_dataset_train_mixed_x6.pkl')

    # BoolLogic.generate_init_expressions_and_save(num_samples=1000000, filename='bool_logic_dataset_train_345_init.pkl')
    # for idx in range(100):
    #     print(f"{idx}", end=' ', flush=True)
    #     BoolLogic.generate_init_expressions_and_save(num_samples=10000, filename=f'datasets_sampling_b/bool_logic_dataset_train_345_grpo_sampling_{idx}.pkl')
    BoolLogic.generate_init_expressions_and_save(num_samples=10000000, filename=f'datasets_sampling_b/bool_logic_dataset_train_345_grpo_sampling.pkl')
    # BoolLogic.generate_dataset_and_save(num_samples=1000000, depth=3, verbose=1, filename='bool_logic_dataset_train.pkl')
    # BoolLogic.generate_dataset_and_save(num_samples=10000, depth=8, verbose=1, filename='bool_logic_dataset_val_d8_v1.pkl')
    # BoolLogic.generate_dataset_and_save(num_samples=1000, depth=9, verbose=1, filename='bool_logic_dataset_val_d9_v1.pkl')
    # BoolLogic.generate_dataset_and_save(num_samples=200, depth=20, verbose=1, filename='bool_logic_dataset_val_d20_v1.pkl')
    # BoolLogic.generate_dataset_and_save(num_samples=100, depth=2, verbose=1, filename='bool_logic_dataset_val_d2_v1.pkl')
    # BoolLogic.generate_dataset_and_save(num_samples=16, depth=1, verbose=1, filename='bool_logic_dataset_val_d1_v1.pkl')
    # dataset = BoolLogic.load_dataset(filename='bool_logic_dataset_train.pkl')