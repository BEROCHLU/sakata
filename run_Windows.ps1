# ログファイルのパスを変数として定義
$logFile = "./result/output3.log"

# バッチファイル生成
python ./utils/slicebatch.py

# 開始時間の記録
$startTime = Get-Date

# 出力ファイルを事前にクリアして現在の日付と時刻を追記
Get-Date | Out-File -FilePath $logFile -Encoding utf8

# batchフォルダ内の全ての.jsonファイルに対してループ
Get-ChildItem -Path ./batch -Filter *.json | ForEach-Object {
    # ファイルの内容を inlinestd.py に渡し、結果を出力ファイルに追記
    Get-Content $_.FullName | python ./src/python/inlinestd.py | Out-File -FilePath $logFile -Append -Encoding utf8
}

# 画像出力
# python ./src/python/plot-batch.py $logFile

# 終了時間の記録
$endTime = Get-Date

# 経過時間の計算（秒単位）
$elapsedTime = $endTime - $startTime

# 経過時間を分と秒に変換
$minutes = [math]::Floor($elapsedTime.TotalMinutes)
$seconds = $elapsedTime.Seconds

# 経過時間をログファイルに追記
"Time: $minutes min $seconds sec." | Out-File -FilePath $logFile -Append -Encoding utf8
