"""Modulo con funciones para visualizar nuestras metrics del entrenamiento."""

from matplotlib import pyplot as plt
from pathlib import Path


def plot_metric_from_dataframe(
    log_dataframe, 
    metric_name: str,
    plots_folder: Path,
):
    "Visualiza una given metric_name del log_df y la guarda en el given plots_folder."
    # Plot backtesting metrics if available
    if metric_name in log_dataframe.columns:
        exception_mask = ~log_dataframe[metric_name].isna()
        plt.figure(figsize=(12, 6))
        plt.plot(log_dataframe['epoch'][exception_mask], log_dataframe[metric_name][exception_mask], label='VaR Exception Rate', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel(f"{metric_name.replace('_', ' ').title()}")
        plt.title(f"VaR {metric_name.replace('_', ' ').title()} over Epochs")
        plt.legend()
        plt.grid(True)
        plot_path = plots_folder / f"train_metrics_{metric_name}.png"
        plt.savefig(plot_path)
        plt.show(block=False)


def plot_metrics(
    log_df,
    plots_folder: Path,
):
    # Plot training and test loss
    plt.figure(figsize=(12, 6))
    plt.plot(log_df['epoch'], log_df['train_loss'], label='Train Loss')
    # Only plot test loss where it's not NaN
    test_loss_mask = ~log_df['test_loss'].isna()
    plt.plot(log_df['epoch'][test_loss_mask], log_df['test_loss'][test_loss_mask], label='Test Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plot_path = plots_folder / f"train_metrics_loss.png"
    plt.savefig(plot_path)
    plt.show(block=False)

    # Plot MAE and MSE
    plt.figure(figsize=(12, 6))
    test_mae_mask = ~log_df['test_mae'].isna()
    test_mse_mask = ~log_df['test_mse'].isna()
    plt.plot(log_df['epoch'][test_mae_mask], log_df['test_mae'][test_mae_mask], label='Test MAE', marker='o')
    plt.plot(log_df['epoch'][test_mse_mask], log_df['test_mse'][test_mse_mask], label='Test MSE', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Test MAE and MSE over Epochs')
    plt.legend()
    plt.grid(True)
    plot_path = plots_folder / f"train_metrics_mse.png"
    plt.savefig(plot_path)
    plt.show(block=False)

    # Plot backtesting metrics if available        
    metrics_list = [
        "exception_rate",
        "kupiec_p",
        "christoffersen_p",
        "expected_shortfall",
    ]
    for metric in metrics_list:
        
        plot_metric_from_dataframe(
            log_dataframe=log_df,
            metric_name=metric,
            plots_folder=plots_folder,
        )
