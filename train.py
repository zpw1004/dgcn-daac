import torch
import numpy as np
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from models import DualGraphFusionGCN
from graph_utils import build_sample_graph, build_cluster_graph
from losses import advanced_jump_penalty
from config import set_seed
import torch.nn as nn

def train_model():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and prepare data
    train_data, scaler, depth_train, cluster_train = build_sample_graph("A.csv",
                                                                        fit_scaler=True)
    test_data, _, depth_test, cluster_test = build_sample_graph("B.csv", scaler=scaler)

    cluster_data_train, Q_train = build_cluster_graph(train_data.x.cpu().numpy())
    cluster_data_test, Q_test = build_cluster_graph(test_data.x.cpu().numpy())

    # Move data to device
    cluster_data_train, cluster_data_test = cluster_data_train.to(device), cluster_data_test.to(device)
    Q_train, Q_test = Q_train.to(device), Q_test.to(device)
    train_data, test_data = train_data.to(device), test_data.to(device)
    depth_train, depth_test = depth_train.to(device), depth_test.to(device)
    cluster_train, cluster_test = cluster_train.to(device), cluster_test.to(device)

    # Initialize model and optimizer
    model = DualGraphFusionGCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_dom = nn.CrossEntropyLoss()

    best_f1 = 0.0
    best_metrics = {}

    print("Start Training...")
    for epoch in range(1, 9001):
        model.train()
        optimizer.zero_grad()

        # Choose alpha value based on training stage
        if epoch <= 4000:
            alpha = 0.0
            stage = "Stage 1"
        else:
            alpha = 2. / (1. + np.exp(-10 * (epoch - 4000) / (9000 - 4000))) - 1
            stage = "Stage 2"

        # Forward pass
        out_src, dom_src, _ = model(train_data, cluster_data_train, Q_train, alpha)
        out_tgt, dom_tgt, _ = model(test_data, cluster_data_test, Q_test, alpha)

        # Unified Loss
        domain_labels = torch.cat([
            torch.zeros(len(train_data.x), dtype=torch.long, device=device),
            torch.ones(len(test_data.x), dtype=torch.long, device=device)
        ], dim=0)
        domain_outputs = torch.cat([dom_src, dom_tgt], dim=0)

        # Calculate losses
        loss_cls = criterion_cls(out_src, train_data.y)
        loss_dom = criterion_dom(domain_outputs, domain_labels)
        loss_jump = advanced_jump_penalty(out_tgt, test_data.edge_index, cluster_test, depth_test)

        # Total loss and backpropagation
        total_loss = loss_cls + loss_dom + 0.1 * loss_jump
        total_loss.backward()
        optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            out_eval, _, _ = model(test_data, cluster_data_test, Q_test, alpha=1.0)
            out_src_eval, _, _ = model(train_data, cluster_data_train, Q_train, alpha=1.0)

            pred_src = out_src_eval.argmax(dim=1)
            acc_src = (pred_src == train_data.y).float().mean().item()

            pred_tgt = out_eval.argmax(dim=1).cpu()
            true_tgt = test_data.y.cpu()
            acc_tgt = (pred_tgt == true_tgt).float().mean().item()
            precision = precision_score(true_tgt, pred_tgt, average='macro', zero_division=0)
            recall = recall_score(true_tgt, pred_tgt, average='macro', zero_division=0)
            f1 = f1_score(true_tgt, pred_tgt, average='macro', zero_division=0)

        print(f"Epoch {epoch:04d} | {stage} | Loss: {total_loss:.4f} | Src Acc: {acc_src:.4f} | "
              f"Tgt Acc: {acc_tgt:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            best_metrics = {
                "epoch": epoch,
                "accuracy": acc_tgt,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
            torch.save(model.state_dict(), "wb_wa_best_fusion_model.pth")
            np.savetxt("wa_wb.txt", pred_tgt.numpy(), fmt='%d')
            cm = confusion_matrix(true_tgt, pred_tgt)
            with open("wa_wb_DGCN_DAAL.pkl", "wb") as f:
                pickle.dump(cm, f)

    # Print final results
    print("\nBest F1 Model Saved at Epoch", best_metrics["epoch"])
    print(f"Best Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"Best Precision: {best_metrics['precision']:.4f}")
    print(f"Best Recall: {best_metrics['recall']:.4f}")
    print(f"Best F1 Score: {best_metrics['f1']:.4f}")


if __name__ == '__main__':
    train_model()