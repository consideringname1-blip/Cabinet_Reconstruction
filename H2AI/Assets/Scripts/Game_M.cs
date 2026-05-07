using UnityEngine;
using UnityEngine.UI;

[DefaultExecutionOrder(-1000)]
public class Game_M : MonoBehaviour
{
    public static Game_M initialize;

    public Text text;
    void Awake()
    {
        initialize = this;
    }

    public void XianShi(string data)
    {
        if (!TryGetMessageRoot(out GameObject messageRoot))
        {
            return;
        }

        messageRoot.SetActive(true);
        text.text = data;
    }

    public void GuanBi()
    {
        if (!TryGetMessageRoot(out GameObject messageRoot))
        {
            return;
        }

        messageRoot.SetActive(false);
        Invoke(nameof(YanXhiGuanBi), 0.5f);
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
