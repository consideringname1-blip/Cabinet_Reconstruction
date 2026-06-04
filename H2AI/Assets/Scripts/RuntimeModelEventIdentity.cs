using Microsoft.MixedReality.Toolkit.Input;
using UnityEngine;

[DisallowMultipleComponent]
public class RuntimeModelEventIdentity : MonoBehaviour, IMixedRealityPointerHandler
{
    [SerializeField] private string modelKey = "";
    [SerializeField] private string taskId = "";
    [SerializeField] private string fbxUrl = "";

    public string ModelKey
    {
        get { return modelKey; }
    }

    public string TaskId
    {
        get { return taskId; }
    }

    public string FbxUrl
    {
        get { return fbxUrl; }
    }

    public void Configure(string newModelKey, string newTaskId, string newFbxUrl)
    {
        modelKey = newModelKey ?? "";
        taskId = newTaskId ?? "";
        fbxUrl = newFbxUrl ?? "";
    }

    public bool TryGetWorldBounds(out Bounds bounds)
    {
        Renderer[] renderers = GetComponentsInChildren<Renderer>(true);
        if (renderers.Length > 0)
        {
            bounds = renderers[0].bounds;
            for (int i = 1; i < renderers.Length; i++)
            {
                bounds.Encapsulate(renderers[i].bounds);
            }
            return true;
        }

        Collider[] colliders = GetComponentsInChildren<Collider>(true);
        if (colliders.Length > 0)
        {
            bounds = colliders[0].bounds;
            for (int i = 1; i < colliders.Length; i++)
            {
                bounds.Encapsulate(colliders[i].bounds);
            }
            return true;
        }

        bounds = new Bounds(transform.position, Vector3.one * 0.1f);
        return false;
    }

    public void OnPointerClicked(MixedRealityPointerEventData eventData)
    {
        ToggleEventPopup();
        if (eventData != null)
        {
            eventData.Use();
        }
    }

    public void OnPointerDown(MixedRealityPointerEventData eventData)
    {
    }

    public void OnPointerDragged(MixedRealityPointerEventData eventData)
    {
    }

    public void OnPointerUp(MixedRealityPointerEventData eventData)
    {
    }

    private void OnMouseDown()
    {
        ToggleEventPopup();
    }

    private void ToggleEventPopup()
    {
        ModelEventDisplay display = ModelEventDisplay.Instance;
        if (display == null)
        {
            Debug.LogWarning("[ModelEvent] ModelEventDisplay is missing from SampleScene/Scripts.");
            return;
        }

        display.ToggleForModel(this);
    }
}
