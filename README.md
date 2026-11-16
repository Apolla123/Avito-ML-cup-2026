# Ранжирование рекомендаций Авито — полный проект

Пайплайн: **признаки → CatBoost (YetiRank) + listwise Transformer → ансамбль → submission.parquet**.
Метрика — NDCG внутри ленты (реализована по правилам соревнования: пессимистичные ничьи, лента без позитивов = 1).

## Что внутри и почему так

| Идея | Зачем |
|---|---|
| **Val идёт в обучение** (`--val-weight 5`) | Val и test собраны одинаково (равномерно перемешанные выдачи), а train смещён исходной позицией (`pos`). Val — самые «честные» данные. |
| **CV по val, фолды по `user_id`** | Модель учится на train + 2/3 val и оценивается на оставшейся трети val; пользователь целиком в одном фолде. Из фолдов получаются OOF-предсказания для честного подбора весов ансамбля, а test-предсказания фолдовых моделей усредняются. |
| **Признаки по истории** (`avito_rank/features.py`) | Тот же товар/категория в истории (счётчики, доли, давность, по типам событий), сходство эмбеддинга кандидата с эмбеддингами истории (max/mean/top3/недавние/взвешенные/по типам событий). У бейзлайна истории нет вообще. |
| **Внутриленточные признаки** | Ранг-перцентиль, z-score, отставание от максимума по f1..f5 и сходствам: ранжирование идёт внутри ленты, абсолютные значения вторичны. |
| **CatBoost YetiRank** | Сильная табличная модель; категория как `cat_feature`; веса лент для val. |
| **Transformer** | Токен пользователя + токены кандидатов, все признаки + вектор товара; listwise softmax-loss (под NDCG) + немного BCE. |
| **Ансамбль** | z-score внутри ленты → взвешенная сумма, веса ищутся по OOF NDCG@30. |

## Установка

Python 3.11. Либо используйте окружение из baseline (там уже есть catboost, torch, polars, click, loguru, pandas), либо:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Запуск

Все команды — из корня проекта; `--data` указывает на распакованный `competition/`.

```bash
# 0. (по желанию) проверить пайплайн на маленьком синтетическом датасете
python -m avito_rank synth --out /tmp/fake
pytest -q tests                      # тесты ядра (признаки, метрика, кэш, ансамбль, сабмит)

# 1. разведка: реальные коды событий, CTR по позициям, доля кандидатов без эмбеддингов
python -m avito_rank eda --data /path/to/competition
#    -> при необходимости поправьте EVENT_TYPES / EVENT_SIM_TYPES в avito_rank/config.py

# 2. признаки (кэшируются в .npz, возобновляемо; один parquet-файл = один .npz)
python -m avito_rank features --data /path/to/competition --cache cache --workers 8
#    для отладки: --limit-parts 1

# 3. быстрый базовый замер: обучение на train, оценка на всём val (как у бейзлайна)
python -m avito_rank train catboost --cache cache --out runs/cb_base --fold -1 --device gpu

# 4. полный цикл: 3 фолда × (CatBoost, NN) → ансамбль → сабмит
DATA=/path/to/competition bash run_all.sh
#    итог: runs/submission.parquet (формат проверяется автоматически), blend.json с весами и OOF-метриками
```

Только CPU: `--device cpu` для CatBoost и `--device cpu` для NN (в `run_all.sh`: `CB_DEVICE=cpu NN_DEVICE=cpu`).
На CPU CatBoost на полном train будет долгим — используйте `--max-train-feeds`.

Отдельно проверить любой сабмит: `python -m avito_rank validate --sample sample_submission.parquet --sub my.parquet`.

## Что подбирать (по убыванию отдачи)

1. `--val-weight` (1 / 3 / 5 / 10): сравнивайте OOF NDCG по фолдам. Это главный рычаг из-за смещения train.
2. `config.py`: `EVENT_TYPES`, `EVENT_SIM_TYPES` — под реальные коды событий (покажет `eda`; важны «контакт/избранное/просмотр»).
3. CatBoost: `--lr 0.05`, `--depth 6..10`, `--loss YetiRankPairwise` / `QueryRMSE`.
4. NN: `--epochs`, `--width 192`, `--layers 4`; несколько сидов (`--seed`) усредняются тем же `blend`.
5. Идеи на будущее: таргет-кодирование `item_id` и пары (user, category) по train+val без утечки между фолдами; IPS-веса по `pos` для train (в `eda` виден CTR по позициям).

## Структура

```
avito_rank/
  features.py        признаки одной ленты (numpy)          build_features.py  параллельный кэш
  data.py            SplitData, кэш, фолды по user_id      metrics.py         NDCG по правилам
  train_catboost.py  CatBoostRanker                         train_nn.py        listwise Transformer
  blend.py           OOF-подбор весов, усреднение фолдов   submission.py      сборка/проверка сабмита
  synth.py           синтетический датасет                  cli.py             команды
tests/test_core.py   проверки ядра без тяжёлых зависимостей
```

## Что проверено, а что нет

Проверено на синтетических данных (`pytest`): признаки против прямого пересчёта, NDCG против прямой реализации (с ничьими), кэш, подвыборки лент, фолды без утечки пользователей, ансамбль и выравнивание сабмита, а также то, что признаки действительно несут обучаемый сигнал.
**Не запускалось** (в среде разработки нет catboost, torch, polars): чтение parquet, `train catboost`, `train nn`, запись parquet-сабмита. Эти части написаны по образцу бейзлайнов; если что-то упадёт — первым делом пришлите трейсбек. Рекомендуемый первый запуск: `synth` → `features` → `train ... --fold -1` на синтетике.
