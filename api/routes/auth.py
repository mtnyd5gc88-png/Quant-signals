from __future__ import annotations
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from api.config import settings

router = APIRouter()


class LoginRequest(BaseModel):
    email: str
    password: str


class UserInfo(BaseModel):
    id: str
    name: str
    email: str
    initials: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserInfo


@router.post("/auth/login", response_model=LoginResponse)
async def login(req: LoginRequest):
    if not req.email or not req.password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email and password are required",
        )

    if not settings.auth_enabled:
        local_part = req.email.split("@")[0].replace(".", " ").replace("_", " ").title()
        parts = local_part.split()
        initials = (parts[0][0] + (parts[-1][0] if len(parts) > 1 else parts[0][-1])).upper()
        return LoginResponse(
            access_token=f"dev-{req.email}",
            token_type="bearer",
            user=UserInfo(
                id="dev-user",
                name=local_part or "Dev User",
                email=req.email,
                initials=initials[:2],
            ),
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication not configured for this deployment",
    )
