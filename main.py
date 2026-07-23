"""
End-to-end pipeline: load data -> train all three models -> evaluate ->
ROC curves -> Grad-CAM.

For interactive exploration (EDA plots, step-by-step results), use the
notebook in notebooks/ instead — this script is the scripted equivalent
for reproducibility.

Usage:
    python main.py
"""

from src import config, data, evaluate, gradcam, models, train


def main():
    # 1. Data
    dataset, train_loader, val_loader, test_loader = data.build_dataloaders()
    class_names = dataset.classes

    # 2. Baseline CNN (frozen ResNet50)
    baseline = models.build_baseline_cnn(num_classes=len(class_names))
    train.train_model(baseline, train_loader, trainable_params=baseline.fc.parameters())
    evaluate.evaluate_model(baseline, test_loader, class_names, label="Baseline CNN")

    # 3. Fine-Tuned CNN
    finetuned = models.build_finetuned_cnn(num_classes=len(class_names))
    train.train_model(finetuned, train_loader)
    evaluate.evaluate_model(finetuned, test_loader, class_names, label="Fine-Tuned CNN")

    # 4. Vision Transformer
    vit = models.build_vision_transformer(num_classes=len(class_names))
    train.train_model(vit, train_loader)
    evaluate.evaluate_model(vit, test_loader, class_names, label="Vision Transformer (ViT-B16)")

    # 5. ROC comparison
    labels_b, probs_b = evaluate.collect_softmax_probs(baseline, test_loader)
    _, probs_f = evaluate.collect_softmax_probs(finetuned, test_loader)
    _, probs_v = evaluate.collect_softmax_probs(vit, test_loader)

    evaluate.plot_roc_comparison(labels_b, {
        "Baseline CNN": probs_b,
        "Fine-Tuned CNN": probs_f,
        "Vision Transformer": probs_v,
    }, num_classes=len(class_names))

    evaluate.plot_per_class_roc(labels_b, probs_f, class_names)

    # 6. Grad-CAM (fine-tuned CNN)
    gradcam.plot_gradcam_all_classes(finetuned, test_loader, class_names, config.DEVICE)

    # 7. Save the best-performing model
    import torch
    torch.save(finetuned.state_dict(), config.MODEL_SAVE_PATH)
    print(f"Saved fine-tuned model to {config.MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
