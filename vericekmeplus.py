import os
import csv
import time
import random # Rastgele bekleme süreleri için bu kütüphaneyi ekliyoruz
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

# ==============================================================================
# 1. AYARLAR
# ==============================================================================
sirketler = [
    # E-Ticaret
    "trendyol", "hepsiburada", "getir", "n11", "ciceksepeti",
    # Kargo
    "yurtici-kargo", "aras-kargo", "mng-kargo", "ptt-kargo",
    # Havayolu
    "thy", "pegasus", "anadolujet",
    # Perakende & Market
    "migros", "a101", "bim-market",
    # Bankacılık
    "ziraat-bankasi", "garanti-bbva", "is-bankasi",
    # Diğer Popüler Şirketler
    "sahibinden-com", "arcelik", "vestel", "spotify"
]

csv_dosya = "toplu_sikayetler_playwright.csv"

# ==============================================================================
# 2. DOSYA YÖNETİMİ
# ==============================================================================
if not os.path.exists(csv_dosya):
    print(f"'{csv_dosya}' dosyası bulunamadı, başlıklar eklenerek oluşturuluyor...")
    with open(csv_dosya, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Başlık", "Şikayet"])

# ==============================================================================
# 3. YARDIMCI FONKSİYONLAR
# ==============================================================================
def get_complaint_links(page, sirket_slug):
    try:
        page.wait_for_selector("a.complaint-layer", timeout=7000)
        links = page.locator("a.complaint-layer").all()
        hrefs = [link.get_attribute("href") for link in links]
        full_links = [
            f"https://www.sikayetvar.com{href}"
            for href in hrefs
            if href and href.startswith(f"/{sirket_slug}/")
        ]
        return full_links
    except PlaywrightTimeoutError:
        print("  - Bu sayfada şikayet linki bulunamadı veya sayfa yüklenemedi.")
        return []

def get_title_and_first_paragraph(page):
    try:
        page.wait_for_selector("h1", timeout=10000)
        title = page.locator("h1").inner_text().strip()
    except PlaywrightTimeoutError:
        title = "❌ Başlık bulunamadı"

    try:
        paragraph = page.locator("xpath=//h1/following::p[1]").inner_text().strip()
    except Exception:
        paragraph = "❌ Paragraf alınamadı"

    return title, paragraph

# ==============================================================================
# 4. ANA İŞLEM BLOĞU
# ==============================================================================
with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    
    # ✨✨✨ YENİ YAKLAŞIM BURADA BAŞLIYOR ✨✨✨
    # 1. Gerçek bir tarayıcının kimliğini taklit ediyoruz (User-Agent).
    # Bu, Cloudflare'a "Ben normal bir Chrome kullanıcısıyım" demenin en basit yoludur.
    context = browser.new_context(
        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
    )
    
    page = context.new_page()

    # Her bir şirket için ana döngüyü başlat
    for sirket_slug in sirketler:
        print("\n" + "="*60)
        print(f"🏢 İŞLEM BAŞLADI: '{sirket_slug.upper()}' ŞİRKETİ İÇİN VERİ ÇEKİLİYOR")
        print("="*60)
        
        for sayfa in range(1, 11):
            base_url = f"https://www.sikayetvar.com/{sirket_slug}?page={sayfa}"
            
            print(f"\n📄 Sayfa {sayfa} ({sirket_slug}) taranıyor...")
            try:
                page.goto(base_url, timeout=30000)
                
                # 2. Daha insan benzeri, rastgele beklemeler ekliyoruz.
                # Botlar genellikle sabit sürelerde bekler, insanlar ise beklemez.
                time.sleep(random.uniform(2.5, 4.5)) # 2.5 ile 4.5 saniye arasında rastgele bir süre bekle

                links = get_complaint_links(page, sirket_slug)
                print(f"🔗 {len(links)} geçerli şikayet linki bulundu.")

                for link in links:
                    try:
                        page.goto(link, timeout=30000)
                        time.sleep(random.uniform(1.5, 3.0)) # Her şikayet sayfası arasında da kısa bir süre bekle
                        
                        title, para = get_title_and_first_paragraph(page)
                        
                        if "❌" not in para:
                            print(f"  ✅ Başarıyla çekildi: {title[:70]}...")
                        else:
                            print(f"  ⚠️ Paragraf çekilemedi: {title[:70]}...")

                        with open(csv_dosya, mode="a", encoding="utf-8", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([title, para])
                    except Exception as e:
                        print(f"  ❌ HATA (Link işlenirken: {link}): {str(e)}")
            except PlaywrightTimeoutError:
                 print(f"  ❌ KRİTİK HATA: Sayfa yüklenirken zaman aşımına uğradı. Muhtemelen Cloudflare engeline takıldı.")
                 print("  ➡️ Bir sonraki şirkete geçiliyor...")
                 break
            except Exception as e:
                print(f"  ❌ KRİTİK HATA (Sayfa {sayfa} açılamadı): {str(e)}")
                break

    print("\n\n🎉 Tüm işlemler bitti. Tarayıcı kapatılıyor.")
    browser.close()