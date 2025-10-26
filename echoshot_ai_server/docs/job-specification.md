# Job Message Specification (v2)

이 문서는 **비디오 처리 파이프라인**에서 사용되는 Job 메시지의 구조를 정의한다.  
모든 Job은 AWS SQS 메시지 형태로 전달되며, 서버에서 `domain/job.py` 모델로 변환되어 처리된다.

---

## 1. Overview

Job은 다음 두 가지 필드를 반드시 포함해야 한다:

- `member_id`: 작업 요청자의 회원 ID (필수)
- `source_s3_key`: 원본 비디오 파일의 S3 경로 (필수)

그 외의 필드는 처리 타입, 파라미터, 콜백 URL, 재시도 관리 등을 포함한다.

---

## 2. JSON Schema
```json
{
  "$id": "https://example.com/video-job.schema.json",
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Video Processing Job",
  "type": "object",
  "required": [
    "job_id",
    "member_id",
    "task_type",
    "source_s3_key",
    "callback_url"
  ],
  "properties": {
    "job_id": {
      "type": "string",
      "description": "Job의 고유 식별자 (UUID)"
    },
    "member_id": {
      "type": "string",
      "description": "작업 요청자 회원 식별자 (필수)"
    },
    "task_type": {
      "type": "string",
      "enum": ["upscale", "subtitle", "audio_extract"],
      "description": "수행될 작업의 타입"
    },
    "source_s3_key": {
      "type": "string",
      "description": "S3 상의 원본 영상 경로"
    },
    "parameters": {
      "type": "object",
      "description": "작업 설정값 (예: scale_factor, position 등)",
      "default": {}
    },
    "callback_url": {
      "type": "string",
      "description": "처리 완료 후 콜백을 받을 Spring 서버의 API URL"
    },
    "priority": {
      "type": "string",
      "enum": ["low", "medium", "high"],
      "description": "작업 우선순위 (선택 필드)"
    },
    "retry_count": {
      "type": "integer",
      "minimum": 0,
      "description": "현재 재시도 횟수 (SQS 재전송 관리용)"
    },
    "submitted_at": {
      "type": "string",
      "format": "date-time",
      "description": "Job 생성 시각 (ISO 8601 형식)"
    },
    "metadata": {
      "type": "object",
      "description": "사용자 정의 메타데이터 (예: 프로젝트명, 태그 등)"
    },
    "version": {
      "type": "string",
      "description": "스키마 버전 (기본값 v2)",
      "default": "v2"
    }
  }
}
```

---

## 3. Example Payload
```json
{
  "version": "v2",
  "job_id": "job-123e4567-e89b-12d3-a456-426614174000",
  "member_id": "user-458",
  "task_type": "upscale",
  "source_s3_key": "videos/user-458/original.mp4",
  "parameters": {
    "scale_factor": 2,
    "model": "RealESRGAN"
  },
  "callback_url": "https://backend.mysite.com/api/video-jobs/callback",
  "priority": "high",
  "retry_count": 0,
  "submitted_at": "2025-10-26T06:30:00Z",
  "metadata": {
    "project": "upscale-demo",
    "client_ip": "203.0.113.42"
  }
}
```
---

## 4. Field Description Summary

| Field         | Type             | Required | Description                                   |
|---------------|------------------|----------|-----------------------------------------------|
| job_id        | string           | Yes      | Job 고유 식별자                                    |
| member_id     | string           | Yes      | 요청자 회원 ID                                     |
| task_type     | string(enum)     | Yes      | upscaling, subtitle, audio extraction 등 작업 종류 |
| source_s3_key | string           | Yes      | S3 내 원본 영상 경로                                 |
| parameters    | object           | No       | 작업 파라미터                                       |
| callback_url  | string           | Yes      | 결과 콜백 받을 API 엔드포인트                            |
| priority      | string(enum)     | No       | `low`, `medium`, `high`                       |
| retry_count   | integer          | No       | 현재 재시도 횟수                                     |
| submitted_at  | string(datetime) | No       | Job 생성 타임스탬프                                  |
| metadata      | object           | No       | 사용자 정의 추가 정보                                  |
| version       | string           | No       | 메시지 스키마 버전 (기본값: v2)                          |

---

## 5. 확장 규칙

1. **하위 호환 방식(version 필드 도입)**
    - 기존 v1 메시지도 수용 가능하도록 기본값 처리.

2. **TaskType 확장 가능**
    - `TaskFactory`에 새로운 Task 클래스 등록으로 손쉽게 확장.

3. **Member 기반 분리 저장**
    - S3 업로드 시 `{member_id}` 토폴로지 유지:
      ```
      videos/{member_id}/input/
      processed/{member_id}/{task_type}/
      ```

4. **메타데이터 확장**
    - 자유로운 구조를 가지며, 프로젝트/환경별 커스텀 정보를 담을 수 있음.

---

## 6. Version History

| Version | Date    | Notes                                                                |
|---------|---------|----------------------------------------------------------------------|
| v1      | 2025-01 | 기본 Job 필드(`job_id`, `task_type`, `source_s3_key`, `callback_url`) 정의 |
| v2      | 2025-10 | `member_id` 필드 추가, 메타데이터 및 버전 관리 필드 포함                               |

---

_Last updated: 2025-10-26_