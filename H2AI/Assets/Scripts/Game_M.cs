using System.Collections;
using System.Collections.Generic;
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

    // Start is called before the first frame update
    void Start()
    {
        initialize = this;
    }
    public void XianShi(string data)
    {
        if (text == null || text.transform == null || text.transform.parent == null)
        {
            return;
        }

        text.transform.parent.gameObject.SetActive(true);
        text.text = data;
    }

    public void GuanBi()
    {
        if (text == null || text.transform == null || text.transform.parent == null)
        {
            return;
        }

        text.transform.parent.gameObject.SetActive(false);
        Invoke(nameof(YanXhiGuanBi), 0.5f);
    }
    public void YanXhiGuanBi()
    {
        if (text == null || text.transform == null || text.transform.parent == null)
        {
            return;
        }

        text.transform.parent.gameObject.SetActive(false);
    }
    // Update is called once per frame
    void Update()
    {
        
    }
}
