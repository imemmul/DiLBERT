import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    df_clean = df.copy()
    
    df_clean['hate'] = df_clean['label'].map({'h': 1, 'nh': 0}) # binary hate, non-hate
    
    target_map = {'p': 0, 'e': 1, 'r': 2} # numeric mapping of target categories
    df_clean['target'] = df_clean['target'].str.lower().str.strip()
    
    # non-hate labels have no target
    df_clean['target'] = (
        df_clean['target']
        .map(target_map)
        .where(df_clean['target'].isin(target_map.keys()))
    )
    df_clean['target'] = df_clean['target'].fillna(-100).astype(int)
    # df_clean['target'] = df_clean['target'].fillna(-100).astype(int) # filling non-hate labels with 0, so that instead predicting h or nh, we only need to predict the target
    
    invalid_hate_mask = (df_clean['hate'] == 1) & (df_clean['target'] == -100)
    df_clean.loc[invalid_hate_mask, 'hate'] = 0
    return df_clean


def validate_dataset(df):
    assert set(df['hate'].unique()).issubset({0, 1}), f"Invalid hate labels: {df['hate'].unique()}"
    
    valid_targets = {-100, 0, 1, 2}
    invalid_targets = set(df['target'].unique()) - valid_targets
    assert not invalid_targets, f"Invalid targets detected: {invalid_targets}"
    
    nh_mask = df['hate'] == 0
    assert (df.loc[nh_mask, 'target'] == -100).all(), "Non-hate samples have invalid targets"
    
    assert not df['text'].isna().any(), "NaN in sentence column"
    assert not df['hate'].isna().any(), "NaN in hate column"
    assert not df['target'].isna().any(), "NaN in target column"
    
    print("All dataset validation checks passed!")

def prepare_loaders(df, tokenizer, batch_size=16):
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['hate'])
    
    train_dataset = TurkishHateSpeechDataset(
        train_df['text'].values,
        train_df['hate'].values,
        train_df['target'].values,
        tokenizer
    )
    
    test_dataset = TurkishHateSpeechDataset(
        test_df['text'].values,
        test_df['hate'].values,
        test_df['target'].values,
        tokenizer
    )

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), \
           DataLoader(test_dataset, batch_size=batch_size)

import torch
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
import numpy as np

class TurkishHateSpeechDataset(Dataset):
    def __init__(self, texts, hate_labels, target_labels, tokenizer, max_len=128):
        self.texts = texts
        self.hate_labels = hate_labels
        self.target_labels = target_labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'hate_labels': torch.tensor(self.hate_labels[idx], dtype=torch.float),
            'target_labels': torch.tensor(self.target_labels[idx], dtype=torch.long)
        }

def evaluate_model(model, dataloader, device):
    model.eval()
    hate_probs = []
    hate_preds = []
    true_hate = []
    target_probs = []
    target_preds = []
    true_target = []
    target_mask = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            hate_logits, target_logits = model(input_ids, attention_mask)
            # target_logits = model(input_ids, attention_mask)
            
            # Hate predictions
            batch_probs = torch.sigmoid(hate_logits.squeeze()).cpu().numpy()
            batch_preds = (batch_probs > 0.5).astype(int)
            
            hate_probs.extend(batch_probs)
            hate_preds.extend(batch_preds)
            true_hate.extend(batch['hate_labels'].cpu().numpy())
            
            # Target predictions
            batch_target_probs = torch.softmax(target_logits, dim=1).cpu().numpy()
            batch_target_preds = np.argmax(batch_target_probs, axis=1)
            
            target_probs.extend(batch_target_probs)
            target_preds.extend(batch_target_preds)
            true_target.extend(batch['target_labels'].cpu().numpy())
            target_mask.extend(batch['target_labels'].cpu().numpy() != -100)

    return {
        'true_hate': np.array(true_hate),
        'pred_hate': np.array(hate_preds),
        'hate_probs': np.array(hate_probs),
        'target_probs': np.array(target_probs),
        'true_target': np.array(true_target)[np.array(target_mask)],
        'pred_target': np.array(target_preds)[np.array(target_mask)],
        'target_mask': np.array(target_mask)
    }

class TurkishHateBERT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained("dbmdz/bert-base-turkish-cased")
        self.hate_head = torch.nn.Linear(768, 1)
        self.target_head = torch.nn.Linear(768, 3)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.hate_head(pooled_output), self.target_head(pooled_output)
    

# class TurkishHateBERT_allinone(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.bert = AutoModel.from_pretrained("dbmdz/bert-base-turkish-cased")
#         self.target_head = torch.nn.Linear(768, 4)

#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert(input_ids, attention_mask=attention_mask)
#         pooled_output = outputs.pooler_output
#         return self.target_head(pooled_output)


def train_model(model, train_loader, test_loader, device, epochs=20, lr=2e-5):
    optimizer = AdamW(model.parameters(), lr=lr)
    hate_criterion = torch.nn.BCEWithLogitsLoss()
    target_criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
    best_f1 = 0
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            hate_labels = batch['hate_labels'].to(device)
            target_labels = batch['target_labels'].to(device)
            
            target_logits = model(input_ids, attention_mask)
            
            # Calculate losses
            # hate_loss = hate_criterion(hate_logits.squeeze(), hate_labels)
            
            loss = target_criterion(
                target_logits,
                target_labels
            )
                
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # val_metrics = evaluate_model(model, test_loader, device)
        print(f"\nEpoch {epoch+1}/{epochs}")
        # print(f"Train Loss: {total_loss/len(train_loader):.4f}")
        # print("\nValidation Metrics:")
        # print(classification_report(
        #     val_metrics['true_hate'], 
        #     val_metrics['pred_hate'], 
        #     target_names=['Non-Hate', 'Hate']
        # ))
        # print("\nTarget Classification (Hate Cases Only):")
        # print(classification_report(
        #     val_metrics['true_target'], 
        #     val_metrics['pred_target'], 
        #     target_names=['Politics', 'Ethnicity', 'Race']
        # ))

    return model

def predict(text, model, tokenizer, device):
    encoding = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(device)
    
    model.eval()
    with torch.no_grad():
        hate_logits, target_logits = model(encoding['input_ids'], encoding['attention_mask'])
    
    hate_prob = torch.sigmoid(hate_logits).item()
    target_probs = torch.softmax(target_logits, dim=1).cpu().numpy()[0]
    
    return {
        'hate_probability': hate_prob,
        'target_probabilities': {
            'politics': target_probs[0],
            'ethnicity': target_probs[1],
            'religious': target_probs[2]
        }
    }

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt
if __name__ == "__main__":
    df = pd.read_csv("./dataset.csv")
    df = df.drop(columns=['annotators'])
    df_clean = preprocess_data(df)
    print(df_clean['target'].value_counts())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
    train_loader, test_loader = prepare_loaders(df_clean, tokenizer)
    model = TurkishHateBERT().to(device)
    cp = torch.load("hatebert_model_vanilla.pth")
    model.load_state_dict(cp)
    # # model = train_model(model, train_loader, test_loader, device)
    results = evaluate_model(model, test_loader, device)
    print(f"Classification Report:\n{classification_report(results['true_hate'], results['pred_hate'])}")
    print(f"Target Classification Report:\n{classification_report(results['true_target'], results['pred_target'])}")
    correct_hate_mask = np.array(results['true_hate']) == 1
    filtered_true_target = np.array(results['true_target'])[correct_hate_mask[results['target_mask']]]
    filtered_pred_target = np.array(results['pred_target'])[correct_hate_mask[results['target_mask']]]
    print(f"Filtered Target Classification Report:\n{classification_report(filtered_true_target, filtered_pred_target, target_names=['Politics', 'Ethnicity', 'Religious'])}")
    cm_target = confusion_matrix(filtered_true_target, filtered_pred_target)
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm_target, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Politics', 'Ethnicity', 'Religious'],
        yticklabels=['Politics', 'Ethnicity', 'Religious']
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix — Target Classification')
    plt.tight_layout()
    plt.savefig("target_confusion_matrix.png")
    # torch.save(model.state_dict(), "hatebert_model.pth")
    all_true_target = results['true_target']                 # shape (M,)
    all_target_probs = results['target_probs'][results['target_mask']]  # now also shape (M, 3)

    # 2) Binarize the true labels
    from sklearn.preprocessing import label_binarize
    classes = [0, 1, 2]
    y_true_bin = label_binarize(all_true_target, classes=classes)  # shape (M, 3)

    # 3) Use the filtered probs as the scores
    y_score = all_target_probs  # shape (M, 3)

    # 4) Compute per-class ROC/AUC
    from sklearn.metrics import roc_curve, auc
    fpr = {}; tpr = {}; roc_auc = {}
    for i, cls in enumerate(classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 5) Plot
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 7))
    for i in classes:
        plt.plot(fpr[i], tpr[i], lw=2,
                label=f"Class {i} ROC (AUC = {roc_auc[i]:.2f})")
    plt.plot([0,1], [0,1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('One-vs-Rest ROC Curves — Target Classification')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("roc_target.png")
    sample = "Ermeni lobisi her işi kontrol ediyor, onlara güven olmaz."
    print(predict(sample, model, tokenizer, device))

