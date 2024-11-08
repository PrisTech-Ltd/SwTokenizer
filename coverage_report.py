# report_generator.py
import matplotlib.pyplot as plt
from datasets import load_dataset
from swahili_tokenizer import SwahiliTokenizer


def generate_report():
    # Load the Swahili dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.sw")
    tokenizer = SwahiliTokenizer()


    # Tokenize each entry in the dataset
    for i, entry in enumerate(ds['train']['text']):
        tokenizer.tokenize(entry)
        # if i > 300:
        #     break

    # Calculate the success rate
    success_count = tokenizer.success.total()
    failure_count = tokenizer.failure.total()
    total = tokenizer.success.total() + tokenizer.failure.total()
    success_rate = success_count / total * 100 if total > 0 else 0

    # Generate report
    with open("morphemes.csv", "w") as f:
        f.write(f"{total=}\n")
        for item, count in tokenizer.success.items():
            f.write(f"{item},{count}\n")

    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Total Successful Tokenizations: {success_count}")
    print(f"Total Failed Tokenizations: {failure_count}")

    # Plotting the results
    labels = ['Success', 'Failure']
    sizes = [success_count, failure_count]
    colors = ['#4CAF50', '#F44336']

    plt.figure(figsize=(8, 6))
    plt.bar(labels, sizes, color=colors)
    plt.title('Morphological Analyzer Success Rate')
    plt.xlabel('Result Type')
    plt.ylabel('Count')
    plt.ylim(0, max(sizes) + 5)

    # Show top 50 failures
    print("\nTop 50 Failures:")
    with open("failures.txt", "w") as f:
        f.write(f"{failure_count=}\n")
        for item, count in tokenizer.failure.most_common(50):
            print(f"{item}: {count}")
            f.write(f"{item},{100*count/failure_count:.2f}%\n")

    # Show most frequent morphemes
    print("\nTop 300 success:")
    with open("successes.txt", "w") as f:
        f.write(f"{success_count=}\n")
        for item, count in tokenizer.success.most_common(300):
            print(f"{item}: {count}")
            f.write(f"{item},{100*count/success_count:.2f}%\n")


    plt.savefig('success_rate_report.png')
    plt.show()

if __name__ == "__main__":
    generate_report()