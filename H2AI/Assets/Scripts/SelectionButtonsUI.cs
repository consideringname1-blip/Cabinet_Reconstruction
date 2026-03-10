using System.Collections;
using System.Collections.Generic;
using System;
using UnityEngine;

public class SelectionButtonsUI : MonoBehaviour
{
    [SerializeField] private SelectionBoxController selectionBoxController;

    public event Action ConfirmClicked;
    public event Action CancelClicked;

    public void OnConfirmClicked()
    {
        if (selectionBoxController == null)
        {
            Debug.LogWarning("SelectionButtonsUI: selectionBoxController is not assigned.");
            ConfirmClicked?.Invoke();
            return;
        }

        Debug.Log($"[Confirm] Clicked.");

        ConfirmClicked?.Invoke();
    }

    public void OnCancelClicked()
    {
        Debug.Log("[Cancel] Clicked.");
        CancelClicked?.Invoke();
    }
}