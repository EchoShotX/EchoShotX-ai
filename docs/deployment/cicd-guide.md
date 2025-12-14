# CI/CD 가이드

## 개요

EchoShotX AI Worker 서비스는 GitHub Actions를 통한 CI/CD 파이프라인을 사용합니다. 코드 변경사항이 `main`/`master` 브랜치에 푸시되면 자동으로 테스트, 빌드, 배포가 실행됩니다.

## CI/CD 파이프라인 흐름

```mermaid
graph LR
    A[코드 푸시] --> B[GitHub Actions 트리거]
    B --> C[코드 체크아웃]
    C --> D[테스트 실행]
    D --> E{테스트 통과?}
    E -->|실패| F[배포 중단]
    E -->|성공| G[배포 패키지 생성]
    G --> H[S3 업로드]
    H --> I[CodeDeploy 트리거]
    I --> J[EC2 배포]
    J --> K[Docker Compose 빌드]
    K --> L[컨테이너 실행]
    L --> M[Health Check]
    M --> N{Health Check 통과?}
    N -->|실패| O[롤백]
    N -->|성공| P[배포 완료]
```

## GitHub Actions 워크플로우

### 워크플로우 파일 위치
`.github/workflows/deploy.yml`

### 주요 단계

1. **코드 체크아웃**
   - 소스 코드 다운로드
   - Git 정보 추출 (커밋 SHA, 브랜치 등)

2. **Python 환경 설정**
   - Python 3.10 설치
   - 의존성 캐싱

3. **테스트 실행**
   - `pytest` 실행
   - 테스트 커버리지 수집 (옵션)

4. **배포 패키지 생성**
   - 소스 코드, 스크립트, docker-compose.yml 포함
   - 서브모듈 포함

5. **S3 업로드**
   - 배포 패키지를 S3에 업로드

6. **CodeDeploy 배포 트리거**
   - AWS CLI를 통한 배포 생성
   - 배포 상태 모니터링

## GitHub Secrets 설정

### 필수 Secrets

다음 Secrets를 GitHub 저장소에 설정해야 합니다:

#### AWS 인증 정보

1. **AWS_ACCESS_KEY_ID** (필수)
   - **설명**: AWS 접근 키 ID
   - **용도**: AWS 서비스 (S3, CodeDeploy) 접근
   - **예시**: `AKIAIOSFODNN7EXAMPLE`
   - **설정 위치**: GitHub 저장소 → Settings → Secrets and variables → Actions

2. **AWS_SECRET_ACCESS_KEY** (필수)
   - **설명**: AWS 시크릿 접근 키
   - **용도**: AWS 서비스 인증
   - **예시**: `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY`
   - **보안**: 절대 공개하지 않음

#### AWS 리소스 정보

3. **AWS_REGION** (선택, 기본값: ap-northeast-2)
   - **설명**: AWS 리전
   - **용도**: 모든 AWS 서비스 요청의 리전 지정 (S3, CodeDeploy)
   - **예시**: `ap-northeast-2`
   - **기본값**: 설정하지 않으면 `ap-northeast-2` 사용

#### CodeDeploy 설정

4. **CODE_DEPLOY_APPLICATION_NAME** (필수)
   - **설명**: CodeDeploy 애플리케이션 이름
   - **용도**: 배포 대상 애플리케이션 지정
   - **예시**: `EchoShotWorker`
   - **확인 방법**: AWS 콘솔 → CodeDeploy → Applications

5. **CODE_DEPLOY_DEPLOYMENT_GROUP** (필수)
   - **설명**: CodeDeploy 배포 그룹 이름
   - **용도**: 배포 대상 인스턴스 그룹 지정
   - **예시**: `production`
   - **확인 방법**: AWS 콘솔 → CodeDeploy → Applications → Deployment groups

#### CodeDeploy 배포용 S3 버킷 (선택)

6. **CODEDEPLOY_S3_BUCKET** (선택)
   - **설명**: CodeDeploy 배포 패키지를 저장할 S3 버킷 이름
   - **용도**: 배포 패키지 업로드 및 CodeDeploy가 접근할 버킷
   - **예시**: `codedeploy-echoshot-apnortheast2` 또는 `my-codedeploy-bucket`
   - **기본값**: 설정하지 않으면 자동으로 `codedeploy-echoshot-{리전}` 형식으로 생성
   - **주의사항**:
     - 버킷이 이미 존재해야 함 (자동 생성되지 않음)
     - CodeDeploy가 접근할 수 있는 권한이 있어야 함
     - 버킷 정책에서 CodeDeploy 서비스 접근 허용 필요

#### GitHub 서브모듈 접근 (Private 서브모듈 사용 시)

7. **PAT_TOKEN** (선택, private 서브모듈 사용 시)
   - **설명**: GitHub Personal Access Token (PAT)
   - **용도**: Private 서브모듈 접근
   - **예시**: `ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
   - **언제 필요한가?**
     - ✅ **필요한 경우**: 다른 조직/저장소의 private 서브모듈
     - ✅ **필요한 경우**: 같은 조직이지만 `GITHUB_TOKEN` 권한이 부족한 경우
     - ❌ **불필요한 경우**: 같은 조직 내 private 서브모듈이고 워크플로우 권한이 충분한 경우
   - **생성 방법**: 
     1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
     2. "Generate new token (classic)" 클릭
     3. Note: "EchoShotX CI/CD" (설명)
     4. Expiration: 적절한 기간 설정 (예: 90일, 1년)
     5. 권한 선택: `repo` (전체 저장소 접근) 체크
     6. "Generate token" 클릭
     7. 토큰 생성 후 **즉시 복사** (다시 볼 수 없음)
   - **설정 방법**:
     1. GitHub 저장소 → Settings → Secrets and variables → Actions
     2. "New repository secret" 클릭
     3. Name: `PAT_TOKEN`
     4. Value: 생성한 Personal Access Token 붙여넣기
     5. "Add secret" 클릭
   - **보안 주의사항**:
     - 토큰은 절대 공개하지 않음
     - 정기적으로 토큰 갱신 권장
     - 최소 권한 원칙 (필요한 저장소만 접근 가능하도록)
   - **워크플로우 동작**:
     - `PAT_TOKEN`이 설정되어 있으면 우선 사용
     - `PAT_TOKEN`이 없으면 기본 `GITHUB_TOKEN` 사용 (자동 fallback)
     - 워크플로우 코드: `token: ${{ secrets.PAT_TOKEN || secrets.GITHUB_TOKEN }}`

### Secrets 설정 방법

1. GitHub 저장소로 이동
2. **Settings** → **Secrets and variables** → **Actions** 클릭
3. **New repository secret** 버튼 클릭
4. Name과 Value 입력
5. **Add secret** 클릭

### Secrets 확인

워크플로우에서 사용되는 Secrets 목록:

```yaml
# .github/workflows/deploy.yml에서 사용
env:
  AWS_REGION: ${{ secrets.AWS_REGION || 'ap-northeast-2' }}
  CODE_DEPLOY_APPLICATION_NAME: ${{ secrets.CODE_DEPLOY_APPLICATION_NAME }}
  CODE_DEPLOY_DEPLOYMENT_GROUP: ${{ secrets.CODE_DEPLOY_DEPLOYMENT_GROUP }}

steps:
  - name: Checkout code
    uses: actions/checkout@v4
    with:
      submodules: recursive
      token: ${{ secrets.PAT_TOKEN || secrets.GITHUB_TOKEN }}

  - name: Configure AWS credentials
    uses: aws-actions/configure-aws-credentials@v4
    with:
      aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
      aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
      aws-region: ${{ env.AWS_REGION }}
```

**참고**: 
- `PAT_TOKEN`이 설정되어 있으면 사용하고, 없으면 기본 `GITHUB_TOKEN` 사용
- 같은 조직 내 private 서브모듈: `GITHUB_TOKEN`으로 가능 (워크플로우 권한 설정 필요)
- 다른 조직/저장소의 private 서브모듈: `PAT_TOKEN` 필수

### Secrets 체크리스트

배포 전 다음 Secrets가 모두 설정되어 있는지 확인:

- [ ] `AWS_ACCESS_KEY_ID`
- [ ] `AWS_SECRET_ACCESS_KEY`
- [ ] `AWS_REGION` (선택)
- [ ] `CODE_DEPLOY_APPLICATION_NAME`
- [ ] `CODE_DEPLOY_DEPLOYMENT_GROUP`
- [ ] `CODEDEPLOY_S3_BUCKET` (선택, 특정 버킷 사용 시)
- [ ] `PAT_TOKEN` (private 서브모듈 사용 시, 필요시에만 설정)

## Docker Compose 사용

이 프로젝트는 Docker Compose를 사용하여 EC2에서 직접 빌드 및 실행합니다.

### Docker Compose 파일

프로젝트 루트에 `docker-compose.yml` 파일이 포함되어 있으며, 다음을 포함합니다:

- Dockerfile 기반 빌드
- 환경 변수 파일(.env.prod) 사용
- 볼륨 마운트 설정
- 네트워크 설정
- 로깅 설정

### 배포 프로세스

1. **소스 코드 배포**: GitHub Actions에서 소스 코드를 S3에 업로드
2. **CodeDeploy**: EC2에 소스 코드 배포
3. **Docker Compose 빌드**: EC2에서 `docker-compose up --build` 실행
4. **컨테이너 실행**: 빌드된 이미지로 컨테이너 시작

### EC2 요구사항

EC2 인스턴스에 다음이 설치되어 있어야 합니다:

- Docker
- Docker Compose
- CodeDeploy Agent

`before_install.sh` 스크립트에서 자동으로 설치를 확인하고 필요시 설치합니다.

### Secrets 설정 방법

1. GitHub 저장소로 이동
2. Settings → Secrets and variables → Actions
3. "New repository secret" 클릭
4. 이름과 값을 입력
5. "Add secret" 클릭

## IAM 권한 설정

### GitHub Actions용 IAM 사용자/역할

다음 권한이 필요합니다:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:CreateBucket",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::codedeploy-echoshot-*",
        "arn:aws:s3:::codedeploy-echoshot-*/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "codedeploy:CreateDeployment",
        "codedeploy:GetApplication",
        "codedeploy:GetApplicationRevision",
        "codedeploy:GetDeployment",
        "codedeploy:GetDeploymentConfig",
        "codedeploy:RegisterApplicationRevision"
      ],
      "Resource": "*"
    }
  ]
}
```

**참고**: ECR 관련 권한은 더 이상 필요하지 않습니다. Docker Compose를 사용하여 EC2에서 직접 빌드하므로 ECR 접근 권한이 필요 없습니다.

## 워크플로우 트리거

### 자동 트리거

#### Push to main/master
```yaml
on:
  push:
    branches:
      - main
      - master
```

#### Pull Request
```yaml
on:
  pull_request:
    branches:
      - main
      - master
```
- Pull Request는 테스트만 실행 (배포하지 않음)

### 수동 트리거

```yaml
on:
  workflow_dispatch:
    inputs:
      environment:
        description: 'Deployment environment'
        required: true
        default: 'production'
        type: choice
        options:
          - production
```

GitHub Actions UI에서 수동으로 워크플로우를 실행할 수 있습니다.

## 배포 환경

### 배포 설정

- **Production**: `main`/`master` 브랜치 푸시 시 자동 배포
- **Pull Request**: 테스트만 실행 (배포하지 않음)

### 환경 변수 주입

워크플로우에서 환경 변수를 설정할 수 있습니다:

```yaml
env:
  AWS_REGION: ${{ secrets.AWS_REGION || 'ap-northeast-2' }}
  CODE_DEPLOY_APPLICATION_NAME: ${{ secrets.CODE_DEPLOY_APPLICATION_NAME }}
  CODE_DEPLOY_DEPLOYMENT_GROUP: ${{ secrets.CODE_DEPLOY_DEPLOYMENT_GROUP }}
```

## CodeDeploy 배포 트리거

### 배포 생성

```bash
aws deploy create-deployment \
  --application-name $CODE_DEPLOY_APPLICATION_NAME \
  --deployment-group-name $CODE_DEPLOY_DEPLOYMENT_GROUP \
  --s3-location bucket=$S3_BUCKET,key=$S3_KEY,bundleType=zip
```

### 배포 상태 확인

```bash
aws deploy get-deployment \
  --deployment-id $DEPLOYMENT_ID
```

## 롤백 전략

### 자동 롤백

1. **Health Check 실패 시**
   - CodeDeploy가 자동으로 이전 버전으로 롤백
   - Health check 스크립트에서 실패 반환

2. **배포 실패 시**
   - GitHub Actions에서 배포 실패 감지
   - 이전 Docker 이미지 태그로 재배포

### 수동 롤백

1. **CodeDeploy 콘솔**
   - 배포 이력에서 이전 배포 선택
   - "Rollback" 클릭

2. **GitHub Actions**
   - 이전 커밋으로 되돌리기
   - 자동 배포 트리거

## 모니터링 및 알림

### GitHub Actions 알림

- 워크플로우 실행 상태를 GitHub에서 확인
- 실패 시 이메일 알림 (설정 시)

### CodeDeploy 알림

- SNS를 통한 배포 상태 알림
- CloudWatch 알람 설정

## 문제 해결

### 워크플로우 실패

1. **로그 확인**
   - GitHub Actions 탭에서 로그 확인
   - 각 단계별 로그 확인

2. **권한 확인**
   - AWS 인증 정보 확인
   - IAM 권한 확인

3. **네트워크 문제**
   - S3 접근 확인
   - 인터넷 연결 확인

### 배포 실패

자세한 내용은 [troubleshooting.md](./troubleshooting.md) 참조

## 최적화 팁

### 1. 캐싱 활용

- Docker 레이어 캐싱
- Python 의존성 캐싱
- 빌드 아티팩트 캐싱

### 2. 병렬 실행

- 테스트와 빌드를 병렬로 실행
- 여러 환경에 동시 배포 (옵션)

### 3. 조건부 실행

- 변경된 파일에 따라 테스트/빌드 스킵
- 특정 경로 변경 시에만 배포

## 보안 고려사항

### Secrets 관리

- Secrets는 절대 코드에 하드코딩하지 않음
- 민감한 정보는 GitHub Secrets 사용
- Secrets는 암호화되어 저장됨

### IAM 최소 권한

- 필요한 권한만 부여
- 정기적으로 권한 검토
- 불필요한 권한 제거

### 이미지 보안

- 보안 스캔 수행 (옵션)
- 취약점 스캔 도구 사용
- 최신 베이스 이미지 사용

## 참고 자료

- [GitHub Actions 문서](https://docs.github.com/en/actions)
- [AWS CodeDeploy 문서](https://docs.aws.amazon.com/codedeploy/)
- [Docker Compose 문서](https://docs.docker.com/compose/)

