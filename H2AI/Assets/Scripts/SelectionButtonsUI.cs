using System;
using UnityEngine;

public class SelectionButtonsUI : MonoBehaviour
{
    public event Action<bool> ConfirmClicked;
    public event Action CancelClicked;

    public void OnConfirmClicked()
    {
        ConfirmClicked?.Invoke(false);
    }

    /// <summary>
    /// Confirmation-window entry for replacing the matched object's model revision.
    /// Assign the force-rebuild confirmation button to this method in the Unity editor.
    /// </summary>
    public void OnForceRebuildAndUploadClicked()
    {
        ConfirmClicked?.Invoke(true);
    }

    public void OnCancelClicked()
    {
        CancelClicked?.Invoke();
    }
}
