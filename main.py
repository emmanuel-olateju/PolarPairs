import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='PolarPairs Training')

    parser.add_argument('--mode', type=str, default='single',
        choices=['single', 'contrastive']
        )
    
    parser.add_argument('--task', type=str, default='subtask-1',
                        choices=['subtask-1', 'subtask-2', 'subtask-3', 'combined'])
    
    parser.add_argument('--lr', type=float, default=1E-3)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=3)

    parser.add_argument('--experiment-version', type=str, default='v0.0')
    parser.add_argument('--experiment-baseline', type=str, default='None')
    parser.add_argument('--experiment-description', type=str, default='Baseline')

    parser.add_argument('--teacher-model', type=str, default='microsoft/deberta-v3-small')
    parser.add_argument('--student-model', type=str, default='microsoft/deberta-v3-small')

    return parser.parse_args()


def main():
    pass