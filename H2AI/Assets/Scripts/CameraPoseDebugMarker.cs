using UnityEngine;

/// <summary>
/// Duplicates the attached GameObject and places debug markers for server poses.
/// </summary>
[DisallowMultipleComponent]
public class CameraPoseDebugMarker : MonoBehaviour
{
    private static CameraPoseDebugMarker _instance;
    public static CameraPoseDebugMarker Instance
    {
        get
        {
            if (_instance == null)
            {
                _instance = FindObjectOfType<CameraPoseDebugMarker>(true);
            }
            return _instance;
        }
        private set
        {
            _instance = value;
        }
    }

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

    private void OnDestroy()
    {
        if (_instance == this)
        {
            _instance = null;
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
        SetMarkerVisible(_cameraMarker, true);

        _modelMarker.transform.SetPositionAndRotation(modelPosition, modelRotation);
        _modelMarker.name = modelMarkerName;
        SetMarkerVisible(_modelMarker, true);

        Debug.Log(
            $"[CameraPoseDebugMarker] Camera marker pos={cameraPosition}, rot={cameraRotation.eulerAngles}; " +
            $"model marker pos={modelPosition}, rot={modelRotation.eulerAngles}"
        );
    }

    public void PlaceCameraMarker(Vector3 cameraPosition, Quaternion cameraRotation)
    {
        EnsureMarkers();
        if (_cameraMarker == null)
        {
            Debug.LogWarning("[CameraPoseDebugMarker] Failed to create camera marker.");
            return;
        }

        _cameraMarker.transform.SetPositionAndRotation(cameraPosition, cameraRotation);
        _cameraMarker.name = cameraMarkerName;
        SetMarkerVisible(_cameraMarker, true);

        Debug.Log(
            $"[CameraPoseDebugMarker] Camera marker pos={cameraPosition}, rot={cameraRotation.eulerAngles}"
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
        SetMarkerVisible(_arucoMarker, true);

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
        SetMarkerVisible(_modelMarker, true);

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

    private void SetMarkerVisible(GameObject marker, bool visible)
    {
        if (marker == null)
        {
            return;
        }

        marker.SetActive(visible);
        foreach (Transform child in marker.GetComponentsInChildren<Transform>(true))
        {
            child.gameObject.SetActive(visible);
        }
        foreach (Renderer renderer in marker.GetComponentsInChildren<Renderer>(true))
        {
            renderer.enabled = visible;
        }
    }
}
