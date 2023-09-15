# Dev_HumanPoseEstimation
Web APIベースの人物姿勢推定システムの開発 for ジム

## 背景
+ 技術で役に立つ
+ お金のかからない姿勢推定システムの需要
+ 定量的評価による判断根拠が重要

## 目的
+ 姿勢推定システムの無料提供
+ IT技術(Mobile/Web/インフラ)を用いた課題解決の実経験を積む

### ロードマップ
| マイルストン | 内容 | 達成状況 |
| :-- | :-- | :-- |
| 0合目 | Dockerコンテナでの各モジュールの展開 | ◯ |
| 1合目 | ローカル環境でブラウザとサーバの疎通確認 | ◯ |
| 2合目 | 画像データ送信, レスポンス確認 | ◯ |
| 3合目 | 姿勢推定モデルの簡易実装(主要サービス) | ◯ | 
| 4合目 | AWS: EC2インスタンスへの展開 | 実行中 |
| 5合目 | バックエンド: DB設計, データフロー設計, WebAPIマイクロサービス設計 | - |
| 6合目 | フロントエンド: UI設計, React(Next.js)の実装 | - |
| 7合目 | バックエンド: 基幹部をNext.jsとDjangoでハイブリッド実装 | - |
| 8合目 | バックエンド: アカウント機能, 認証機能 | - |
| 9合目 | 姿勢推定モデルの多様化 | - |

### '23/9 機能要件(構想段階)
+ アカウント機能
+ 認証機能
+ DB機能
+ 静止画 姿勢推定機能
+ 動画 姿勢推定機能
+ 静止画保存取り出し機能
+ 動画保存取り出し機能
+ Browserアクセス機能
+ Mobileアクセス機能
+ 姿勢推定モデル切り替え機能

## 現状のシステム構成 '23/9

## 3 layers web application
+ Server : RaspberryPi 3 Modwl B+
+ Platform : arm 32bit, Linux(Debian)
+ Container : Docker
+ Frontend : JavaScript, TypeScript
+ Backend : Python (Flask)
+ DL : Tensorflow Lite
+ DB : Redis(KVS)

## 姿勢推定モデル一覧
+ PoseNet
+ MoveNet
+ OpenPose

