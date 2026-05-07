using UnityEngine;

/// <summary>
/// Duplicates the attached GameObject and places debug markers for server poses.
/// </summary>
[DisallowMultipleComponent]
public class CameraPoseDebugMarker : MonoBehaviour
{
    public static CameraPoseDebugMarker Instance { get; private set; }

    [SerializeField] private Transform markerParent;
    [SerializeField] private string cameraMarkerName = "RecordedCameraPoseMarker";
    [SerializeField] private string modelMarkerName = "RecordedFinalModelMarker";
    [SerializeField] private string arucoMarkerName = "RecordedArucoReferenceMarker";
    [SerializeField] private bool hideTemplateOnStart = false;

    private GameObject _cameraMarker;
    private GameObject _modelMarker;
    private GameObject _arucoMarker;

    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(this);
            return;
        }

        Instance = this;
        if (hideTemplateOnStart)
        {
            gameObject.SetActive(false);
        }
    }

    public void PlaceMarkers(
        Vector3 cameraPosition,
        Quaternion cameraRotation,
        Vector3 modelPosition,
        Quaternion modelRotation
    )
    {
        EnsureMarkers();
        if (_cameraMarker == null || _modelMarker == null)
        {
            Debug.LogWarning("[CameraPoseDebugMarker] Failed to create debug markers.");
            return;
        }

        _cameraMarker.transform.SetPositionAndRotation(cameraPosition, cameraRotation);
        _cameraMarker.name = cameraMarkerName;
        _cameraMarker.SetActive(true);

        _modelMarker.transform.SetPositionAndRotation(modelPosition, modelRotation);
        _modelMarker.name = modelMarkerName;
        _modelMarker.SetActive(true);

        Debug.Log(
            $"[CameraPoseDebugMarker] Camera marker pos={cameraPosition}, rot={cameraRotation.eulerAngles}; " +
            $"model marker pos={modelPosition}, rot={modelRotation.eulerAngles}"
        );
    }

    public void PlaceArucoMarker(Vector3 arucoPosition, Quaternion arucoRotation)
    {
        EnsureMarkers();
        if (_arucoMarker == null)
        {
            Debug.LogWarning("[CameraPoseDebugMarker] Failed to create ArUco marker.");
            return;
        }

        _arucoMarker.transform.SetPositionAndRotation(arucoPosition, arucoRotation);
        _arucoMarker.name = arucoMarkerName;
        _arucoMarker.SetActive(true);

        Debug.Log(
            $"[CameraPoseDebugMarker] ArUco marker pos={arucoPosition}, rot={arucoRotation.eulerAngles}"
        );
    }

    public void PlaceModelMarker(Vector3 modelPosition, Quaternion modelRotation)
    {
        EnsureMarkers();
        if (_modelMarker == null)
        {
            Debug.LogWarning("[CameraPoseDebugMarker] Failed to create model marker.");
            return;
        }

        _modelMarker.transform.SetPositionAndRotation(modelPosition, modelRotation);
        _modelMarker.name = modelMarkerName;
        _modelMarker.SetActive(true);

        Debug.Log(
            $"[CameraPoseDebugMarker] Model marker pos={modelPosition}, rot={modelRotation.eulerAngles}"
        );
    }

    private void EnsureMarkers()
    {
        if (_cameraMarker == null)
        {
            _cameraMarker = CreateMarkerClone(cameraMarkerName);
        }

        if (_modelMarker == null)
        {
            _modelMarker = CreateMarkerClone(modelMarkerName);
        }

        if (_arucoMarker == null)
        {
            _arucoMarker = CreateMarkerClone(arucoMarkerName);
        }
    }

    private GameObject CreateMarkerClone(string cloneName)
    {
        GameObject clone = Instantiate(gameObject, markerParent);
        CameraPoseDebugMarker cloneMarker = clone.GetComponent<CameraPoseDebugMarker>();
        if (cloneMarker != null)
        {
            if (Application.isPlaying)
            {
                Destroy(cloneMarker);
            }
            else
            {
                DestroyImmediate(cloneMarker);
            }
        }
        clone.name = cloneName;
        clone.SetActive(false);
        return clone;
    }

    private void SetMarkerActive(GameObject marker, bool active)
    {
        if (marker != null)
        {
            marker.SetActive(active);
        }
    }
}
