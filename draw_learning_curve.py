from matplotlib import pyplot as plt

def extract_data(file_path):
    # Extracts training and validation metrics from a log file.
    # Assumes the log file has lines formatted as:
    # iter 308: loss -0.0019, lr 5.95e-06, rewards 0.4417 (ema 0.5662), acc 0.4583 (ema 0.5348), p_value 0.0949, time 23108.91ms, mfu 0.06%


    iter = []
    loss = []
    rewards = []
    rewards_ema = []
    acc = []
    acc_ema = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('iter'):
                line = line.replace(':', ',')
                parts = line.split(',')
                # print(parts)
                # exit()
                iter.append(int(parts[0].split()[1]))
                loss.append(float(parts[1].split()[1]))
                rewards.append(float(parts[3].split()[1]))
                rewards_ema.append(float(parts[3].split()[3][-7:-1]))
                acc.append(float(parts[4].split()[1]))
                acc_ema.append(float(parts[4].split()[3][-7:-1]))

    return iter, loss, rewards, rewards_ema, acc, acc_ema

def plot_learning_curve(iter, loss, rewards, rewards_ema, acc, acc_ema, save_path=None):
    # Plots the learning curve using matplotlib.
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(iter, loss, label='Loss', color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss over Iterations')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(iter, rewards, label='Rewards', color='green')
    plt.plot(iter, rewards_ema, label='Rewards EMA', color='orange', linestyle='--')
    plt.xlabel('Iteration')
    plt.ylabel('Rewards')
    plt.title('Training Rewards over Iterations')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(iter, acc, label='Accuracy', color='red')
    plt.plot(iter, acc_ema, label='Accuracy EMA', color='purple', linestyle='--')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy over Iterations')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path if save_path else 'learning_curve.png')
    plt.close()

if __name__ == "__main__":
    log_file_path = 'train_log/rope_l2_grpo_34_noimplies_a.log'  # Replace with your log file path
    iter, loss, rewards, rewards_ema, acc, acc_ema = extract_data(log_file_path)
    plot_learning_curve(iter, loss, rewards, rewards_ema, acc, acc_ema, save_path=f'{log_file_path}.png')