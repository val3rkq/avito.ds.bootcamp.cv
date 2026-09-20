# Ориентация текстового кропа: 0/180 градусов

Предсказание `p_180` — вероятности, что кроп текста перевёрнут вверх ногами. Метрика: 1 − Brier.

## Данные

Архив `test.zip` распаковывается **в корень репозитория**, чтобы в корне было:

```
sample_submission.csv
test/images/test_00000.png ... test_19999.png
```

## Запуск через Docker

Для получения результата `outputs/submission.csv` (запуск скрипта, ноутбук не нужен):

```bash
docker compose run --rm baseline
```

Jupyter с ноутбуками на http://localhost:8888 (нужен только для EDA):

```bash
docker compose up -d jupyter
```

## Подход (попытка 1)

Готовый классификатор угла из PaddleOCR (`ch_ppocr_mobile_v2.0_cls`, MobileNetV3, вход 3×48×192) **без дообучения**.
Для длинных строк (w/h > 4) кроп режется до 4 окон шириной 192, логиты усредняются.

Антисимметричный TTA: `z = (logit(x) − logit(rot180(x))) / 2`, `p = σ(z)` --- гарантирует, что `p(x) + p(rot180 x) = 1`.

Open-source компоненты: PaddleOCR / RapidOCR (модель, препроцессинг), onnxruntime, OpenCV.
