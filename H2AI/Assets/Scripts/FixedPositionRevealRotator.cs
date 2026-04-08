using UnityEngine;

public class FixedPositionRevealRotator : MonoBehaviour
{
    [SerializeField] private Vector3 fixedPosition = Vector3.zero;
    [SerializeField] private Vector3 fixedEulerAngles = Vector3.zero;
    [SerializeField] private Vector3 rotationEulerPerSecond = new Vector3(0f, 60f, 0f);
    [SerializeField] private bool useLocalSpace = false;

    private Renderer[] _renderers;
    private bool _isRotating;
    private Vector3 _currentPositionOffset = Vector3.zero;
    private Vector3 _currentEulerOffset = Vector3.zero;

    void Awake()
    {
        _renderers = GetComponentsInChildren<Renderer>(true);
        ResetTransform();
        SetVisible(false);
    }

    void Update()
    {
        if (!_isRotating)
        {
            return;
        }

        if (useLocalSpace)
        {
            transform.Rotate(rotationEulerPerSecond * Time.deltaTime, Space.Self);
        }
        else
        {
            transform.Rotate(rotationEulerPerSecond * Time.deltaTime, Space.World);
        }
    }

    public void ShowRotateAndReset()
    {
        _currentPositionOffset = Vector3.zero;
        _currentEulerOffset = Vector3.zero;
        ResetTransform();
        SetVisible(true);
        _isRotating = true;
    }

    public void ShowRotateAndResetWithOffset(Vector3 positionOffset)
    {
        _currentPositionOffset = positionOffset;
        _currentEulerOffset = Vector3.zero;
        ResetTransform();
        SetVisible(true);
        _isRotating = true;
    }

    public void ShowRotateAndResetWithOffset(Vector3 positionOffset, Vector3 eulerOffset)
    {
        _currentPositionOffset = positionOffset;
        _currentEulerOffset = eulerOffset;
        ResetTransform();
        SetVisible(true);
        _isRotating = true;
    }

    public void ShowAtOffset(Vector3 positionOffset)
    {
        _currentPositionOffset = positionOffset;
        _currentEulerOffset = Vector3.zero;
        ResetTransform();
        SetVisible(true);
        _isRotating = false;
    }

    public void ShowAtOffset(Vector3 positionOffset, Vector3 eulerOffset)
    {
        _currentPositionOffset = positionOffset;
        _currentEulerOffset = eulerOffset;
        ResetTransform();
        SetVisible(true);
        _isRotating = false;
    }

    public void HideAndStop()
    {
        _isRotating = false;
        ResetTransform();
        SetVisible(false);
    }

    void ResetTransform()
    {
        Quaternion baseRotation = Quaternion.Euler(fixedEulerAngles);
        Quaternion targetRotation = baseRotation * Quaternion.Euler(_currentEulerOffset);
        Vector3 targetPosition = fixedPosition + _currentPositionOffset;

        if (useLocalSpace)
        {
            transform.localPosition = targetPosition;
            transform.localRotation = targetRotation;
        }
        else
        {
            transform.position = targetPosition;
            transform.rotation = targetRotation;
        }
    }

    void SetVisible(bool visible)
    {
        foreach (Renderer rendererComponent in _renderers)
        {
            rendererComponent.enabled = visible;
        }
    }
}
