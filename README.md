# qm_portal

wgpuiを用いた、独自フォーマットの量子化学計算管理ツールです。

通常のディレクトリを1つのプロジェクトとして扱い、JSONベースのファイルで分子情報、計算タスク、実行結果を管理します。GUIはこれらのファイルを編集・実行・可視化するための入口になります。

## 目的

- `charge`、`spin`、`atoms`、`bonds` からなる分子情報を再利用する
- 計算手法、汎関数、基底関数、溶媒、`scan`、`freeze` などの条件を明示する
- 入力ファイルと出力ファイルをタスク単位で対応づける
- 外部プログラムに依存する計算結果を再現可能な形で記録する

## プロジェクト構成

プロジェクトは特別なデータベースではなく通常のディレクトリです。Gitでそのまま差分管理・共有できます。

```text
my-project/
  project.toml
  molecules/
  tasks/
  inputs/
  outputs/
  runs/
```

フォーマットは [`docs/format.md`](docs/format.md)、運用は [`docs/workflow.md`](docs/workflow.md)、開発ルールは [`AGENTS.md`](AGENTS.md)、設計判断は [`docs/adr/`](docs/adr/) を参照してください。
