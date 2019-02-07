python src/run_configuration.py --n-trials 5 --dataset Fashion-MNIST --network-type fc --hidden-nodes 1024 --batch-size 100 --epochs 200 --momentum 0.9 --learning-rates 0.02 --n-modules 1 --output-directory 1-M-1024-H --lambda-values 1.0 --seed 42083 --early-stop

python src/run_configuration.py --n-trials 5 --dataset Fashion-MNIST --network-type fc --hidden-nodes 64 --batch-size 100 --epochs 200 --momentum 0.9 --learning-rates 0.32 0.32 0.32 0.32 0.32 0.096 0.096 --n-modules 16 --output-directory 16-M-64-H --lambda-values 0.0 0.5 0.8 0.9 0.95 0.99 1.0 --seed 36734 --early-stop

python src/run_configuration.py --n-trials 5 --dataset Fashion-MNIST --network-type fc --hidden-nodes 16 --batch-size 100 --epochs 200 --momentum 0.9 --learning-rates 1.28 1.28 1.28 1.28 1.28 0.384 0.384 --n-modules 64 --output-directory 64-M-16-H --lambda-values 0.0 0.5 0.8 0.9 0.95 0.99 1.0 --seed 48636 --early-stop

python src/run_configuration.py --n-trials 5 --dataset Fashion-MNIST --network-type fc --hidden-nodes 4 --batch-size 100 --epochs 200 --momentum 0.9 --learning-rates 5.15 5.15 5.15 5.15 5.15 1.536 1.536 --n-modules 256 --output-directory 256-M-4-H --lambda-values 0.0 0.5 0.8 0.9 0.95 0.99 1.0 --seed 44826 --early-stop
