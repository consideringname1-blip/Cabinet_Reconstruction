using System;
using UnityEngine;

public class SelectionButtonsUI : MonoBehaviour
{
    public event Action ConfirmClicked;
    public event Action CancelClicked;

    public void OnConfirmClicked()
    {
        ConfirmClicked?.Invoke();
    }

    public void OnCancelClicked()
    {
        CancelClicked?.Invoke();
    }
}
