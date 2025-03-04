
# 코드 스타일
Structura는 일관된 코드 스타일을 유지하기 위해 아래의 규칙을 따릅니다.

## 1.1 코드 포맷팅
Structura는 `ruff`를 사용하여 코드 스타일을 자동으로 정리합니다.  
PR을 제출하기 전에 반드시 `ruff`를 실행해주세요:

```sh
ruff check .
ruff format .
```

추가적인 설정은 `pyproject.toml`에 정의되어 있습니다.

## 1.2 Docstring 스타일
모든 함수 및 클래스에는 **NumPy 스타일**의 Docstring을 작성해야 합니다.

예제:

```python
def calculate_density(data):
    """
    주어진 데이터의 밀도를 계산합니다.

    Parameters
    ----------
    data : numpy.ndarray
        밀도를 계산할 데이터.

    Returns
    -------
    float
        계산된 밀도 값.
    """
    return data.mean()
```

NumPy 스타일 Docstring에 대한 자세한 내용은 [NumPy Docstring Guide](https://numpydoc.readthedocs.io/en/latest/format.html)를 참고하세요.

---

# 프로젝트 구조
Structura는 최신 Python 프로젝트 구조를 따릅니다.

```
Structura/
│── structura/        # 주요 모듈 (라이브러리 코드)
│── examples/         # 사용 예제 및 데모 코드
│── tests/            # 테스트 코드 (pytest 사용)
│── docs/             # 문서화 파일 (Sphinx 활용 가능)
│── pyproject.toml    # 프로젝트 설정
│── requirements.txt  # 의존성 목록
│── README.md         # 프로젝트 개요 및 사용법
│── CONTRIBUTING.md   # 기여 가이드
```

- `setup.py` 대신 `pyproject.toml`을 사용하여 프로젝트를 관리합니다.
- 모든 새로운 기능은 `examples/` 폴더에 사용 예제를 포함해야 합니다.

---

# 기여 방식
Structura에 기여하는 방법은 다음과 같습니다.

## 3.1 Issue 등록
기능 요청, 버그 리포트는 [GitHub Issues](https://github.com/mmingyeong/Structura/issues)에 등록해주세요.  
등록할 때 다음 형식을 참고해주세요:

```markdown
**설명**:
버그 또는 추가할 기능에 대한 명확한 설명.

**재현 방법**:
버그의 경우, 발생 과정을 단계별로 정리.

**기대하는 동작**:
올바르게 동작하는 경우의 예상 결과.
```

## 3.2 Pull Request (PR) 제출
- 기여하려면 `fork` 후, `feature/{기능명}` 또는 `bugfix/{수정명}` 브랜치를 생성하세요.
- 변경 사항을 반영한 후, PR을 요청해주세요.
- PR 작성 시 반드시 다음 사항을 포함해야 합니다:
  - 변경 사항 설명
  - 관련된 Issue 번호 (있다면)
  - 테스트 결과

---

# 테스트 방법
- 모든 코드 변경 사항은 반드시 테스트를 포함해야 합니다.
- Structura는 `pytest`를 사용하여 테스트를 실행합니다.
- 테스트 실행 방법:

```sh
pytest tests/
```

- 테스트 코드는 `tests/` 디렉토리에 위치해야 합니다.

---

# 라이선스
Structura는 [MIT 라이선스](LICENSE)를 따릅니다.  
코드 기여자는 해당 라이선스를 준수해야 합니다.
```
