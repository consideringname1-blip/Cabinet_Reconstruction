using UnityEngine;
using UnityEngine.UI;

[DefaultExecutionOrder(-1000)]
public class Game_M : MonoBehaviour
{
    public static Game_M initialize;
    private const float DefaultMessageSeconds = 2.5f;

    public Text text;
    void Awake()
    {
        initialize = this;
    }

    public void XianShi(string data)
    {
        Debug.Log("[FRONT_MESSAGE] " + (data ?? ""));
        CancelInvoke(nameof(YanXhiGuanBi));
        if (!TryGetMessageRoot(out GameObject messageRoot))
        {
            return;
        }

        messageRoot.SetActive(true);
        text.text = data;
    }

    public void XianShiForSeconds(string data)
    {
        XianShiForSeconds(data, DefaultMessageSeconds);
    }

    public void XianShiForSeconds(string data, float seconds)
    {
        XianShi(data);
        CancelInvoke(nameof(YanXhiGuanBi));
        Invoke(nameof(YanXhiGuanBi), Mathf.Max(0.1f, seconds));
    }

    public void GuanBi()
    {
        Debug.Log("[FRONT_MESSAGE] close");
        CancelInvoke(nameof(YanXhiGuanBi));
        if (!TryGetMessageRoot(out GameObject messageRoot))
        {
            return;
        }

        messageRoot.SetActive(false);
    }

    private void YanXhiGuanBi()
    {
        if (TryGetMessageRoot(out GameObject messageRoot))
        {
            messageRoot.SetActive(false);
        }
    }

    private bool TryGetMessageRoot(out GameObject messageRoot)
    {
        messageRoot = null;
        if (text == null || text.transform.parent == null)
        {
            return false;
        }

        messageRoot = text.transform.parent.gameObject;
        return true;
    }
}
