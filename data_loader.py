# poor man's data loader
from data_generator_bool import BoolLogic as Dataset, BoolLogicTokenizer as Tokenizer
tokenizer = Tokenizer()
data_train = Dataset.load_dataset(filename='datasets/bool_logic_dataset_train_345_init.pkl') #
data_val = Dataset.load_dataset(filename='datasets/bool_logic_dataset_val_d7_v1.pkl')
data_train['offset'] = 0
data_val['offset'] = 0

def get_batch_val():
    data = data_val

    x_to_stack = []
    y_to_stack = []
    loss_mask_to_stack = []
    for idx in range(batch_size):
        expression_idx = data['offset'] + idx
        tokenized_expression = data['tokenized_expressions'][expression_idx][:]
        if len(tokenized_expression) > block_size:
            # truncate the expression to fit into the block size
            tokenized_expression = tokenized_expression[-block_size:]
            target_expression = tokenized_expression[1:] + [tokenizer.tokens[' ']]  # shift by one for target
        else:
            # pad the expression to fit into the block size
            tokenized_expression += [tokenizer.tokens[' ']] * (block_size - len(tokenized_expression))
            target_expression = tokenized_expression[1:] + [tokenizer.tokens[' ']]  # shift by one for target

        loss_mask = [0] * data['pos_implies'][expression_idx] + \
                    [1] * (len(data['tokenized_expressions'][expression_idx]) - data['pos_implies'][expression_idx]) + \
                    [0] * (block_size - len(data['tokenized_expressions'][expression_idx]) - 1)
        loss_mask_to_stack.append(torch.tensor(loss_mask, dtype=torch.float32))
        x_to_stack.append(torch.from_numpy(np.array(tokenized_expression, dtype=np.int64)))
        y_to_stack.append(torch.from_numpy(np.array(target_expression, dtype=np.int64)))

    x = torch.stack(x_to_stack)
    y = torch.stack(y_to_stack)
    loss_mask = torch.stack(loss_mask_to_stack)
    x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)

    data['offset'] += batch_size
    if data['offset'] + batch_size > len(data['tokenized_expressions']):
        # reset the offset to 0, so we can loop over the dataset
        data['offset'] = 0
        print(f"Resetting {split} dataset offset to 0")

    return x, y, loss_mask


def get_batch_train_ori():
    data = data_train

    len_data = len(data['tokenized_expressions'])

    x_to_stack = []
    y_to_stack = []
    loss_mask_to_stack = []
    for idx in range(batch_size):
        expression_idx = (data['offset'] + idx) % len_data
        tokenized_expression = data['tokenized_expressions'][expression_idx][:]
        if len(tokenized_expression) > block_size:
            # truncate the expression to fit into the block size
            tokenized_expression = tokenized_expression[-block_size:]
            target_expression = tokenized_expression[1:] + [tokenizer.tokens[' ']]  # shift by one for target
        else:
            # pad the expression to fit into the block size
            tokenized_expression += [tokenizer.tokens[' ']] * (block_size - len(tokenized_expression))
            target_expression = tokenized_expression[1:] + [tokenizer.tokens[' ']]  # shift by one for target

        loss_mask = [0] * data['pos_implies'][expression_idx] + \
                    [1] * (len(data['tokenized_expressions'][expression_idx]) - data['pos_implies'][expression_idx]) + \
                    [0] * (block_size - len(data['tokenized_expressions'][expression_idx]) - 1)
        loss_mask_to_stack.append(torch.tensor(loss_mask, dtype=torch.float32))
        x_to_stack.append(torch.from_numpy(np.array(tokenized_expression, dtype=np.int64)))
        y_to_stack.append(torch.from_numpy(np.array(target_expression, dtype=np.int64)))

    x = torch.stack(x_to_stack)
    y = torch.stack(y_to_stack)
    loss_mask = torch.stack(loss_mask_to_stack)
    x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)

    return x, y, loss_mask


def get_batch_train():
    data = data_train

    x_to_stack = []
    y_to_stack = []
    score_to_stack = []

    for idx in range(batch_size):
        expression_idx = data['offset'] + idx
        tokenized_sampled_expressions = data['tokenized_sampled_expressions'][expression_idx][:]

        sub_x_to_stack = []
        sub_y_to_stack = []

        for expr in tokenized_sampled_expressions:
            tokenized_expression = expr[:]
            if len(tokenized_expression) > block_size:
                # truncate the expression to fit into the block size
                tokenized_expression = tokenized_expression[-block_size:]
                target_expression = tokenized_expression[1:] + [tokenizer.tokens[' ']]  # shift by one for target
            else:
                # pad the expression to fit into the block size
                tokenized_expression += [tokenizer.tokens[' ']] * (block_size - len(tokenized_expression))
                target_expression = tokenized_expression[1:] + [tokenizer.tokens[' ']]  # shift by one for target

            sub_x_to_stack.append(tokenized_expression)
            sub_y_to_stack.append(target_expression)

        x_to_stack.append(sub_x_to_stack)
        y_to_stack.append(sub_y_to_stack)
        score_to_stack.append(data['score'][expression_idx])

    x = torch.tensor(x_to_stack, dtype=torch.long, device=device)  # (B, num_answers, T)
    y = torch.tensor(y_to_stack, dtype=torch.long, device=device)  # (B, num_answers, T)
    scores = torch.tensor(score_to_stack, dtype=torch.float32, device=device)  # (B, num_answers)
    x, y, scores = x.to(device), y.to(device), scores.to(device)

    data['offset'] += batch_size
    if data['offset'] + batch_size > len(data['tokenized_sampled_expressions']):
        # reset the offset to 0, so we can loop over the dataset
        data['offset'] = 0
        print(f"Resetting {split} dataset offset to 0") 

    return x, y, scores


def get_batch(split):

    if split == 'train':
        return get_batch_train()
    elif split == 'val':
        return get_batch_val()
    elif split == 'train_ori':
        return get_batch_train_ori()
