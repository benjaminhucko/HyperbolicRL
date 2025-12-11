# Hyperbolic Reinforcement learning
### Experiments for stationary case:
- breakout
```bash
python main.py --geometry euclidean --updates 1 --epochs 300 --distributional --env breakout
python main.py --geometry hyperbolic --updates 1 --epochs 300 --distributional --learn-curvature --env breakout
```
- freeway
```bash
python main.py --geometry euclidean --updates 1 --epochs 300 --distributional --env freeway
python main.py --geometry hyperbolic --updates 1 --epochs 300 --distributional --learn-curvature --env freeway
```
- asterix
```bash
python main.py --geometry euclidean --updates 1 --epochs 300 --distributional --env asterix
python main.py --geometry hyperbolic --updates 1 --epochs 300 --distributional --learn-curvature --env asterix
```
### Experiments for standard RL case:
- breakout
```bash
python main.py --geometry euclidean --updates 300 --epochs 5 --distributional --learn-curvature --analyze --visualize --env breakout
python main.py --geometry hyperbolic --updates 300 --epochs 40 --distributional --learn-curvature --analyze --visualize --env breakout
```
- freeway
- ```bash
python main.py --geometry euclidean --updates 300 --epochs 5 --distributional --learn-curvature --analyze --visualize --env freeway
python main.py --geometry hyperbolic --updates 300 --epochs 40 --distributional --learn-curvature --analyze --visualize --env freeway
```
- asterix
```bash
python main.py --geometry euclidean --updates 300 --epochs 5 --distributional --learn-curvature --analyze --visualize --env asterix
python main.py --geometry hyperbolic --updates 300 --epochs 40 --distributional --learn-curvature --analyze --visualize --env asterix
```

