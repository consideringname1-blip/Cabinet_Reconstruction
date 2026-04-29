using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
/// <summary>
/// 游戏管理
/// </summary>
public class Game_M : MonoBehaviour
{
    public static Game_M initialize;

    public Text text;
    // Start is called before the first frame update
    void Start()
    {
        initialize = this;
    }
    public void XianShi(string data)
    {
        text.transform.parent.gameObject.SetActive(true);
        text.text = data;
    }

    public void GuanBi()
    {
        text.transform.parent.gameObject.SetActive(false);
        Invoke("YanXhiGuanBi", 0.5f);
    }
    public void YanXhiGuanBi()
    {
        text.transform.parent.gameObject.SetActive(false);
    }
    // Update is called once per frame
    void Update()
    {
        
    }
}
