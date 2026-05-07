using UnityEngine;

using Microsoft.MixedReality.Toolkit.UI;

public class SelectionBoxController : MonoBehaviour
{
    private enum ActiveHandle
    {
        None,
        TopLeft,
        BottomRight
    }

    [Header("References")]
    [SerializeField] private RectTransform imageArea;
    [SerializeField] private RectTransform selectionBox;
    [SerializeField] private Transform handleTL;
    [SerializeField] private Transform handleBR;

    [Header("Initialize On Start (normalized, top-left origin)")]
    [SerializeField] private bool initializeOnStart = true;

    [SerializeField, Range(0f, 1f)] private float initLeft = 0.10f;
    [SerializeField, Range(0f, 1f)] private float initTop = 0.10f;
    [SerializeField, Range(0f, 1f)] private float initRight = 0.90f;
    [SerializeField, Range(0f, 1f)] private float initBottom = 0.90f;

    [Header("Minimum Box Size (normalized)")]
    [SerializeField, Range(0.01f, 1f)] private float minWidthNormalized = 0.05f;
    [SerializeField, Range(0.01f, 1f)] private float minHeightNormalized = 0.05f;

    [Header("Optional")]
    [SerializeField] private bool autoFindReferences = true;

    private ObjectManipulator tlManipulator;
    private ObjectManipulator brManipulator;

    private ActiveHandle activeHandle = ActiveHandle.None;
    private bool subscribed = false;

    // 固定两个handle所在的本地Z平面，避免漂出图片平面
    private float tlPlaneZ = 0f;
    private float brPlaneZ = 0f;

    private void Reset()
    {
        imageArea = GetComponent<RectTransform>();

        if (autoFindReferences)
        {
            AutoFindReferences();
        }
    }

    private void Awake()
    {
        if (imageArea == null)
        {
            imageArea = GetComponent<RectTransform>();
        }

        if (autoFindReferences)
        {
            AutoFindReferences();
        }

        tlManipulator = handleTL != null ? handleTL.GetComponent<ObjectManipulator>() : null;
        brManipulator = handleBR != null ? handleBR.GetComponent<ObjectManipulator>() : null;
    }

    private void OnEnable()
    {
        SubscribeManipulatorEvents();
    }

    private void OnDisable()
    {
        UnsubscribeManipulatorEvents();
    }

    private void Start()
    {
        CacheHandlePlaneZ();

        if (initializeOnStart)
        {
            ResetToInitial();
        }
        else
        {
            ConstrainHandles();
            UpdateSelectionVisual();
        }
    }

    private void LateUpdate()
    {
        if (!IsReady())
            return;

        ConstrainHandles();
        UpdateSelectionVisual();
    }

    private bool IsReady()
    {
        return imageArea != null &&
               selectionBox != null &&
               handleTL != null &&
               handleBR != null;
    }

    private void AutoFindReferences()
    {
        if (selectionBox == null)
        {
            Transform t = transform.Find("SelectionBox");
            if (t != null) selectionBox = t.GetComponent<RectTransform>();
        }

        Transform handleRoot = transform.Find("HandleRoot3D");
        if (handleRoot != null)
        {
            if (handleTL == null)
            {
                Transform tl = handleRoot.Find("HandleTL3D");
                if (tl != null) handleTL = tl;
            }

            if (handleBR == null)
            {
                Transform br = handleRoot.Find("HandleBR3D");
                if (br != null) handleBR = br;
            }
        }
    }

    private void SubscribeManipulatorEvents()
    {
        if (subscribed) return;

        if (tlManipulator != null)
        {
            tlManipulator.OnManipulationStarted.AddListener(OnTLManipulationStarted);
            tlManipulator.OnManipulationEnded.AddListener(OnTLManipulationEnded);
        }

        if (brManipulator != null)
        {
            brManipulator.OnManipulationStarted.AddListener(OnBRManipulationStarted);
            brManipulator.OnManipulationEnded.AddListener(OnBRManipulationEnded);
        }

        subscribed = true;
    }

    private void UnsubscribeManipulatorEvents()
    {
        if (!subscribed) return;

        if (tlManipulator != null)
        {
            tlManipulator.OnManipulationStarted.RemoveListener(OnTLManipulationStarted);
            tlManipulator.OnManipulationEnded.RemoveListener(OnTLManipulationEnded);
        }

        if (brManipulator != null)
        {
            brManipulator.OnManipulationStarted.RemoveListener(OnBRManipulationStarted);
            brManipulator.OnManipulationEnded.RemoveListener(OnBRManipulationEnded);
        }

        subscribed = false;
    }

    private void OnTLManipulationStarted(ManipulationEventData data)
    {
        activeHandle = ActiveHandle.TopLeft;
    }

    private void OnTLManipulationEnded(ManipulationEventData data)
    {
        activeHandle = ActiveHandle.None;
        ConstrainHandles();
        UpdateSelectionVisual();
    }

    private void OnBRManipulationStarted(ManipulationEventData data)
    {
        activeHandle = ActiveHandle.BottomRight;
    }

    private void OnBRManipulationEnded(ManipulationEventData data)
    {
        activeHandle = ActiveHandle.None;
        ConstrainHandles();
        UpdateSelectionVisual();
    }

    private void CacheHandlePlaneZ()
    {
        if (handleTL != null)
        {
            tlPlaneZ = GetLocalPointInImageArea(handleTL).z;
        }

        if (handleBR != null)
        {
            brPlaneZ = GetLocalPointInImageArea(handleBR).z;
        }
    }

    [ContextMenu("Reset To Initial")]
    private void ResetToInitial()
    {
        SetFromNormalized(initLeft, initTop, initRight, initBottom);
    }

    /// <summary>
    /// 使用“相对于图片左上角”的归一化坐标初始化/设置框。
    /// left/top/right/bottom 都是 0~1。
    /// top 往下增大，left 往右增大。
    /// </summary>
    private void SetFromNormalized(float left, float top, float right, float bottom)
    {
        if (!IsReady())
            return;

        SanitizeNormalizedRect(ref left, ref top, ref right, ref bottom);

        Vector3 tlLocal = NormalizedToLocal(left, top, tlPlaneZ);
        Vector3 brLocal = NormalizedToLocal(right, bottom, brPlaneZ);

        SetHandleLocalPoint(handleTL, tlLocal);
        SetHandleLocalPoint(handleBR, brLocal);

        ConstrainHandles();
        UpdateSelectionVisual();
    }

    private void SetFromNormalized(Vector2 topLeft, Vector2 bottomRight)
    {
        SetFromNormalized(topLeft.x, topLeft.y, bottomRight.x, bottomRight.y);
    }

    private void SanitizeNormalizedRect(ref float left, ref float top, ref float right, ref float bottom)
    {
        left = Mathf.Clamp01(left);
        top = Mathf.Clamp01(top);
        right = Mathf.Clamp01(right);
        bottom = Mathf.Clamp01(bottom);

        float minW = Mathf.Clamp01(minWidthNormalized);
        float minH = Mathf.Clamp01(minHeightNormalized);

        // 保证 right >= left + minW
        if (right < left + minW)
        {
            right = left + minW;
        }

        // 保证 bottom >= top + minH
        if (bottom < top + minH)
        {
            bottom = top + minH;
        }

        // 超界后整体往回推
        if (right > 1f)
        {
            float overflow = right - 1f;
            right = 1f;
            left = Mathf.Max(0f, left - overflow);
        }

        if (bottom > 1f)
        {
            float overflow = bottom - 1f;
            bottom = 1f;
            top = Mathf.Max(0f, top - overflow);
        }

        // 再保险一次
        left = Mathf.Clamp01(left);
        top = Mathf.Clamp01(top);
        right = Mathf.Clamp(right, left + minW, 1f);
        bottom = Mathf.Clamp(bottom, top + minH, 1f);
    }

    private void ConstrainHandles()
    {
        if (!IsReady())
            return;

        Rect r = imageArea.rect;

        float minWidthLocal = Mathf.Max(1e-5f, minWidthNormalized * r.width);
        float minHeightLocal = Mathf.Max(1e-5f, minHeightNormalized * r.height);

        Vector3 tl = GetLocalPointInImageArea(handleTL);
        Vector3 br = GetLocalPointInImageArea(handleBR);

        tl.z = tlPlaneZ;
        br.z = brPlaneZ;

        // 先各自限制在图片范围内
        tl.x = Mathf.Clamp(tl.x, r.xMin, r.xMax);
        tl.y = Mathf.Clamp(tl.y, r.yMin, r.yMax);

        br.x = Mathf.Clamp(br.x, r.xMin, r.xMax);
        br.y = Mathf.Clamp(br.y, r.yMin, r.yMax);

        // 再根据当前抓的是谁，防止交叉
        switch (activeHandle)
        {
            case ActiveHandle.TopLeft:
                tl.x = Mathf.Clamp(tl.x, r.xMin, br.x - minWidthLocal);
                tl.y = Mathf.Clamp(tl.y, br.y + minHeightLocal, r.yMax);
                break;

            case ActiveHandle.BottomRight:
                br.x = Mathf.Clamp(br.x, tl.x + minWidthLocal, r.xMax);
                br.y = Mathf.Clamp(br.y, r.yMin, tl.y - minHeightLocal);
                break;

            case ActiveHandle.None:
            default:
                // 没有正在拖拽时，也做一次稳定化约束
                tl.x = Mathf.Min(tl.x, br.x - minWidthLocal);
                tl.y = Mathf.Max(tl.y, br.y + minHeightLocal);

                tl.x = Mathf.Clamp(tl.x, r.xMin, r.xMax - minWidthLocal);
                tl.y = Mathf.Clamp(tl.y, r.yMin + minHeightLocal, r.yMax);

                br.x = Mathf.Max(br.x, tl.x + minWidthLocal);
                br.y = Mathf.Min(br.y, tl.y - minHeightLocal);

                br.x = Mathf.Clamp(br.x, r.xMin + minWidthLocal, r.xMax);
                br.y = Mathf.Clamp(br.y, r.yMin, r.yMax - minHeightLocal);
                break;
        }

        SetHandleLocalPoint(handleTL, tl);
        SetHandleLocalPoint(handleBR, br);
    }

    private void UpdateSelectionVisual()
    {
        if (!IsReady())
            return;

        Vector3 tl = GetLocalPointInImageArea(handleTL);
        Vector3 br = GetLocalPointInImageArea(handleBR);

        float left = Mathf.Min(tl.x, br.x);
        float right = Mathf.Max(tl.x, br.x);
        float top = Mathf.Max(tl.y, br.y);
        float bottom = Mathf.Min(tl.y, br.y);

        float width = right - left;
        float height = top - bottom;

        Vector2 center = new Vector2(
            (left + right) * 0.5f,
            (top + bottom) * 0.5f
        );

        selectionBox.sizeDelta = new Vector2(width, height);
        selectionBox.anchoredPosition = center;
    }

    private Vector3 GetLocalPointInImageArea(Transform target)
    {
        return imageArea.InverseTransformPoint(target.position);
    }

    private void SetHandleLocalPoint(Transform target, Vector3 localPoint)
    {
        target.position = imageArea.TransformPoint(localPoint);
    }

    private Vector3 NormalizedToLocal(float x01, float y01, float z)
    {
        Rect r = imageArea.rect;

        float x = Mathf.Lerp(r.xMin, r.xMax, x01);
        float y = Mathf.Lerp(r.yMax, r.yMin, y01); // top-left origin，向下增大

        return new Vector3(x, y, z);
    }

    private Vector2 LocalToNormalized(Vector3 localPoint)
    {
        Rect r = imageArea.rect;

        float x01 = Mathf.InverseLerp(r.xMin, r.xMax, localPoint.x);
        float y01 = Mathf.InverseLerp(r.yMax, r.yMin, localPoint.y); // top-left origin

        return new Vector2(x01, y01);
    }

    /// <summary>
    /// 返回 TL / BR 两个点的归一化坐标（相对于图片左上角）
    /// </summary>
    public void GetNormalizedTLBR(out Vector2 topLeft, out Vector2 bottomRight)
    {
        Vector3 tl = GetLocalPointInImageArea(handleTL);
        Vector3 br = GetLocalPointInImageArea(handleBR);

        float left = Mathf.Min(tl.x, br.x);
        float right = Mathf.Max(tl.x, br.x);
        float top = Mathf.Max(tl.y, br.y);
        float bottom = Mathf.Min(tl.y, br.y);

        topLeft = LocalToNormalized(new Vector3(left, top, 0f));
        bottomRight = LocalToNormalized(new Vector3(right, bottom, 0f));
    }

    public void PrepareForReuse()
    {
        CacheHandlePlaneZ();
        ResetToInitial();
        ConstrainHandles();
        UpdateSelectionVisual();
    }
}
