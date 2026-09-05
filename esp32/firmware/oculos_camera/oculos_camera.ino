// Firmware dos oculos: ESP32-S3-CAM + OV5640.
//
// Faz uma coisa so -- cria a propria rede Wi-Fi e serve a camera em MJPEG no
// mesmo formato que o esp32/mock_esp32_cam.py serve. Do lado do app nada muda:
// so o endereco.
//
// A rede e criada PELA PLACA (SoftAP). Nao depende de roteador, de internet
// nem de nada instalado no lugar -- que e o requisito do projeto: funcionar em
// qualquer lugar, inclusive onde nao ha Wi-Fi nenhum. O celular entra nesta
// rede e fala direto com a placa.
//
// Gravar precisa da placa; compilar, nao. Ver esp32/README.md.

#include <WiFi.h>
#include "esp_camera.h"
#include "esp_http_server.h"
#include "esp_system.h"

// Qual placa. Os pinos vem do camera_pins.h do proprio core do ESP32, copiado
// para esta pasta sem alteracao -- digitar numero de pino a mao e como se
// perdem tardes inteiras achando que a camera queimou.
#define CAMERA_MODEL_ESP32S3_EYE
#include "camera_pins.h"

// --- A rede que a placa cria ----------------------------------------------
static const char *REDE_NOME = "VisuAll-Oculos";
// WPA2 exige 8 caracteres no minimo. Sem senha, qualquer um por perto entra na
// rede e ve a camera.
static const char *REDE_SENHA = "visuall2026";
// Fixo pra o endereco no app nunca mudar: http://192.168.4.1/stream
static const IPAddress IP_PLACA(192, 168, 4, 1);
static const IPAddress MASCARA(255, 255, 255, 0);

// --- O formato, que TEM de bater com o app --------------------------------
//
// Nao mexa nestas linhas sem mexer no MjpegReader do app.
//
// A fronteira vem ANTES de cada quadro, nao depois. O CameraWebServer de
// fabrica manda depois; navegador tolera, mas e o contrario do que diz o
// formato multipart, e o app foi testado contra este aqui.
//
// O Content-Length nao e enfeite: o leitor do app exige. Sem ele nao ha como
// saber onde termina o JPEG, porque os bytes da imagem podem conter qualquer
// coisa, inclusive algo parecido com a fronteira.
#define FRONTEIRA "123456789000000000000987654321"
static const char *TIPO_STREAM = "multipart/x-mixed-replace;boundary=" FRONTEIRA;
static const char *CABECALHO_PARTE =
    "--" FRONTEIRA "\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

static httpd_handle_t servidor = NULL;

// --- Camera ---------------------------------------------------------------

static bool inicia_camera() {
  camera_config_t cfg = {};
  cfg.ledc_channel = LEDC_CHANNEL_0;
  cfg.ledc_timer = LEDC_TIMER_0;
  cfg.pin_d0 = Y2_GPIO_NUM;
  cfg.pin_d1 = Y3_GPIO_NUM;
  cfg.pin_d2 = Y4_GPIO_NUM;
  cfg.pin_d3 = Y5_GPIO_NUM;
  cfg.pin_d4 = Y6_GPIO_NUM;
  cfg.pin_d5 = Y7_GPIO_NUM;
  cfg.pin_d6 = Y8_GPIO_NUM;
  cfg.pin_d7 = Y9_GPIO_NUM;
  cfg.pin_xclk = XCLK_GPIO_NUM;
  cfg.pin_pclk = PCLK_GPIO_NUM;
  cfg.pin_vsync = VSYNC_GPIO_NUM;
  cfg.pin_href = HREF_GPIO_NUM;
  cfg.pin_sccb_sda = SIOD_GPIO_NUM;
  cfg.pin_sccb_scl = SIOC_GPIO_NUM;
  cfg.pin_pwdn = PWDN_GPIO_NUM;
  cfg.pin_reset = RESET_GPIO_NUM;
  cfg.xclk_freq_hz = 20000000;
  cfg.frame_size = FRAMESIZE_QVGA;    // 320x240, o mesmo do mock e do app
  cfg.pixel_format = PIXFORMAT_JPEG;  // comprimido no proprio chip da camera
  cfg.jpeg_quality = 12;              // 0-63, MENOR e melhor
  cfg.grab_mode = CAMERA_GRAB_LATEST;
  cfg.fb_location = CAMERA_FB_IN_PSRAM;
  cfg.fb_count = 2;

  if (!psramFound()) {
    // Sem PSRAM cabe um buffer so, e na RAM interna. A N16R8 tem 8MB, entao
    // cair aqui e sinal de PSRAM desligada na hora de compilar -- vale mais
    // avisar que seguir devagar em silencio.
    Serial.println("AVISO: PSRAM nao encontrada; confira a placa escolhida na IDE");
    cfg.fb_location = CAMERA_FB_IN_DRAM;
    cfg.fb_count = 1;
    cfg.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  }

  esp_err_t erro = esp_camera_init(&cfg);
  if (erro != ESP_OK) {
    Serial.printf("esp_camera_init falhou: 0x%x (%s)\n", erro, esp_err_to_name(erro));
    Serial.println("  0x105 (ESP_ERR_NOT_FOUND): a camera nao respondeu.");
    Serial.println("  Quase sempre e o modelo de placa errado la em cima,");
    Serial.println("  ou o cabo flat mal encaixado / invertido.");
    return false;
  }

  sensor_t *s = esp_camera_sensor_get();
  // O PID diz QUAL sensor respondeu. Se a camera inicia mas a imagem sai
  // preta, esta linha separa "pino errado" de "sensor diferente do esperado".
  Serial.printf("sensor detectado: PID 0x%02x  (OV5640=0x56, OV2640=0x26, OV3660=0x36)\n",
                s->id.PID);

  // Sem espelho e sem virar de cabeca pra baixo.
  //
  // Nos oculos prontos quem os usa e o ouvinte, e a camera aponta pra quem
  // sinaliza: ela ve outra pessoa de frente, que e a mesma geometria da camera
  // traseira do celular. O app nao espelha essa, e o reconhecimento ja foi
  // conferido assim.
  s->set_hmirror(s, 0);
  s->set_vflip(s, 0);
  return true;
}

// --- HTTP -----------------------------------------------------------------

static esp_err_t pagina(httpd_req_t *req) {
  static const char *HTML =
      "<!doctype html><meta charset=\"utf-8\"><title>VisuAll Oculos</title>"
      "<body style=\"background:#111;color:#eee;font-family:sans-serif;text-align:center\">"
      "<h2>VisuAll &mdash; camera dos oculos</h2>"
      "<img src=\"/stream\" style=\"width:640px;image-rendering:pixelated\">"
      "<p>O app consome <code>/stream</code></p>";
  httpd_resp_set_type(req, "text/html; charset=utf-8");
  return httpd_resp_send(req, HTML, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t stream(httpd_req_t *req) {
  esp_err_t res = httpd_resp_set_type(req, TIPO_STREAM);
  if (res != ESP_OK) return res;
  httpd_resp_set_hdr(req, "Cache-Control", "no-cache");

  char cabecalho[96];
  while (true) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("esp_camera_fb_get devolveu nada; encerrando o stream");
      return ESP_FAIL;
    }

    size_t n = snprintf(cabecalho, sizeof(cabecalho), CABECALHO_PARTE, (unsigned)fb->len);
    res = httpd_resp_send_chunk(req, cabecalho, n);
    if (res == ESP_OK) res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
    if (res == ESP_OK) res = httpd_resp_send_chunk(req, "\r\n", 2);

    esp_camera_fb_return(fb);

    // Erro aqui e o celular tendo fechado a conexao. E o fim normal de um
    // stream, nao um defeito: quem sai do modo oculos fecha o socket.
    if (res != ESP_OK) return res;
  }
}

static void inicia_servidor() {
  httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
  cfg.server_port = 80;
  // O stream nunca termina e ocupa um handler pra sempre. Com folga aqui, a
  // pagina de teste continua respondendo enquanto o app recebe imagem.
  cfg.max_open_sockets = 4;

  if (httpd_start(&servidor, &cfg) != ESP_OK) {
    Serial.println("httpd_start falhou");
    return;
  }
  httpd_uri_t rota_raiz = {"/", HTTP_GET, pagina, NULL};
  httpd_uri_t rota_stream = {"/stream", HTTP_GET, stream, NULL};
  httpd_register_uri_handler(servidor, &rota_raiz);
  httpd_register_uri_handler(servidor, &rota_stream);
}

// --- Ligar ----------------------------------------------------------------

static void conta_por_que_reiniciou() {
  // Brownout e o defeito classico desta montagem: o pico de corrente do Wi-Fi
  // derruba a alimentacao e a placa reinicia sozinha, parecendo travamento ou
  // firmware ruim. Dito aqui, vira uma linha no monitor serial em vez de uma
  // tarde de investigacao. A correcao e o capacitor de 330uF, nao desligar o
  // detector de brownout.
  esp_reset_reason_t r = esp_reset_reason();
  Serial.printf("motivo do boot: %d", (int)r);
  if (r == ESP_RST_BROWNOUT) {
    Serial.print("  <<< BROWNOUT: falta corrente. Confira o capacitor de 330uF");
    Serial.print(" e os 5,0V do MT3608");
  }
  Serial.println();
}

void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println();
  conta_por_que_reiniciou();

  if (!inicia_camera()) {
    Serial.println("sem camera; nada a servir");
    return;
  }

  WiFi.mode(WIFI_AP);
  WiFi.softAPConfig(IP_PLACA, IP_PLACA, MASCARA);
  // max_connection 1: so o celular. Cada cliente a mais divide a banda de um
  // radio que ja e o gargalo.
  WiFi.softAP(REDE_NOME, REDE_SENHA, 1 /* canal */, 0 /* visivel */, 1 /* clientes */);
  // Sem isto o radio dorme entre pacotes e o stream engasga.
  WiFi.setSleep(false);

  inicia_servidor();

  Serial.println();
  Serial.printf("rede...: %s   senha: %s\n", REDE_NOME, REDE_SENHA);
  Serial.printf("no app.: http://%s/stream\n", WiFi.softAPIP().toString().c_str());
  Serial.printf("no nav.: http://%s/\n", WiFi.softAPIP().toString().c_str());
}

void loop() {
  // Nada aqui: o servidor HTTP roda na propria tarefa. Um loop vazio com delay
  // evita ocupar CPU a toa e deixa o watchdog em paz.
  delay(1000);
}
