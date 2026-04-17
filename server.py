#!/usr/bin/env python3

"""
Shopify MCP Server — Full Admin API access via FastMCP.

Provides tools for managing products, orders, customers, collections,
inventory, fulfillments, pages, themes, files, blogs, metaobjects,
and publications through the Shopify Admin REST API.

Token Management:
- Uses client_credentials grant to auto-generate and refresh tokens
- Set SHOPIFY_CLIENT_ID + SHOPIFY_CLIENT_SECRET (recommended for OAuth apps)
- Falls back to static SHOPIFY_ACCESS_TOKEN if client credentials not set
"""

import json
import os
import logging
import time
import asyncio
from typing import Optional, List, Dict, Any
from enum import Enum

import httpx
from pydantic import BaseModel, Field, ConfigDict, field_validator
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SHOPIFY_STORE = os.environ.get("SHOPIFY_STORE", "")
SHOPIFY_TOKEN = os.environ.get("SHOPIFY_ACCESS_TOKEN", "")
SHOPIFY_CLIENT_ID = os.environ.get("SHOPIFY_CLIENT_ID", "")
SHOPIFY_CLIENT_SECRET = os.environ.get("SHOPIFY_CLIENT_SECRET", "")
API_VERSION = os.environ.get("SHOPIFY_API_VERSION", "2024-10")
TOKEN_REFRESH_BUFFER = int(os.environ.get("TOKEN_REFRESH_BUFFER", "1800"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("shopify_mcp")

PORT = int(os.environ.get("PORT", "8000"))
MCP_TRANSPORT = os.environ.get("MCP_TRANSPORT", "streamable-http")

mcp = FastMCP("shopify_mcp", host="0.0.0.0", port=PORT, json_response=True)

# ---------------------------------------------------------------------------
# Token Manager
# ---------------------------------------------------------------------------

class TokenManager:
    def __init__(self, store, client_id, client_secret, static_token="", refresh_buffer=1800):
        self._store = store
        self._client_id = client_id
        self._client_secret = client_secret
        self._static_token = static_token
        self._refresh_buffer = refresh_buffer
        self._access_token: str = ""
        self._expires_at: float = 0.0
        self._lock = asyncio.Lock()
        self._use_client_credentials = bool(client_id and client_secret)

        if self._use_client_credentials:
            logger.info("Token mode: client_credentials (auto-refresh enabled)")
        elif static_token:
            logger.info("Token mode: static SHOPIFY_ACCESS_TOKEN (no auto-refresh)")
            self._access_token = static_token
            self._expires_at = float("inf")
        else:
            logger.warning("No credentials configured.")

    @property
    def is_expired(self) -> bool:
        if not self._access_token:
            return True
        return time.time() >= (self._expires_at - self._refresh_buffer)

    async def get_token(self) -> str:
        if not self.is_expired:
            return self._access_token
        async with self._lock:
            if not self.is_expired:
                return self._access_token
            if self._use_client_credentials:
                await self._refresh_token()
            elif not self._access_token:
                raise RuntimeError("No valid token available.")
        return self._access_token

    async def force_refresh(self) -> str:
        if not self._use_client_credentials:
            raise RuntimeError("Cannot refresh — using a static token.")
        async with self._lock:
            await self._refresh_token()
        return self._access_token

    async def _refresh_token(self) -> None:
        url = f"https://{self._store}.myshopify.com/admin/oauth/access_token"
        logger.info("Refreshing Shopify access token...")
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url,
                data={"grant_type": "client_credentials", "client_id": self._client_id, "client_secret": self._client_secret},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=15.0,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Token refresh failed ({resp.status_code}). Check SHOPIFY_CLIENT_ID and SHOPIFY_CLIENT_SECRET.")
            data = resp.json()
            self._access_token = data["access_token"]
            expires_in = data.get("expires_in", 86399)
            self._expires_at = time.time() + expires_in
            logger.info(f"Token refreshed. Expires in {expires_in}s.")


token_manager = TokenManager(
    store=SHOPIFY_STORE,
    client_id=SHOPIFY_CLIENT_ID,
    client_secret=SHOPIFY_CLIENT_SECRET,
    static_token=SHOPIFY_TOKEN,
    refresh_buffer=TOKEN_REFRESH_BUFFER,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _base_url() -> str:
    return f"https://{SHOPIFY_STORE}.myshopify.com/admin/api/{API_VERSION}"

async def _headers() -> dict:
    token = await token_manager.get_token()
    return {"X-Shopify-Access-Token": token, "Content-Type": "application/json"}

async def _request(method, path, params=None, body=None, _retried=False) -> dict:
    if not SHOPIFY_STORE:
        raise RuntimeError("Missing SHOPIFY_STORE environment variable.")
    url = f"{_base_url()}/{path}"
    headers = await _headers()
    async with httpx.AsyncClient() as client:
        resp = await client.request(method, url, headers=headers, params=params, json=body, timeout=30.0)
        if resp.status_code == 401 and not _retried and token_manager._use_client_credentials:
            await token_manager.force_refresh()
            return await _request(method, path, params=params, body=body, _retried=True)
        resp.raise_for_status()
        if resp.status_code == 204:
            return {}
        return resp.json()

def _error(e: Exception) -> str:
    if isinstance(e, httpx.HTTPStatusError):
        status = e.response.status_code
        try:
            detail = e.response.json()
        except Exception:
            detail = e.response.text[:500]
        messages = {
            401: "Authentication failed — check your token.",
            403: "Permission denied — missing API scopes.",
            404: "Resource not found.",
            422: f"Validation error: {json.dumps(detail)}",
            429: "Rate-limited — wait and retry.",
        }
        return messages.get(status, f"Shopify API error {status}: {json.dumps(detail)}")
    if isinstance(e, httpx.TimeoutException):
        return "Request timed out."
    if isinstance(e, RuntimeError):
        return str(e)
    return f"Unexpected error: {type(e).__name__}: {e}"

def _fmt(data: Any) -> str:
    return json.dumps(data, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════
# PRODUCTS
# ═══════════════════════════════════════════════════════════════════════════

class ListProductsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    limit: Optional[int] = Field(default=50, ge=1, le=250)
    status: Optional[str] = Field(default=None, description="active, archived, draft")
    product_type: Optional[str] = Field(default=None)
    vendor: Optional[str] = Field(default=None)
    collection_id: Optional[int] = Field(default=None)
    since_id: Optional[int] = Field(default=None)
    fields: Optional[str] = Field(default=None)

@mcp.tool(name="shopify_list_products")
async def shopify_list_products(params: ListProductsInput) -> str:
    """List products from the Shopify store with optional filters."""
    try:
        p: Dict[str, Any] = {"limit": params.limit}
        for field in ["status", "product_type", "vendor", "collection_id", "since_id", "fields"]:
            val = getattr(params, field)
            if val is not None:
                p[field] = val
        data = await _request("GET", "products.json", params=p)
        products = data.get("products", [])
        return _fmt({"count": len(products), "products": products})
    except Exception as e:
        return _error(e)

class GetProductInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_id: int = Field(..., description="The Shopify product ID")

@mcp.tool(name="shopify_get_product")
async def shopify_get_product(params: GetProductInput) -> str:
    """Retrieve a single product by ID, including all variants and images."""
    try:
        data = await _request("GET", f"products/{params.product_id}.json")
        return _fmt(data.get("product", data))
    except Exception as e:
        return _error(e)

class CreateProductInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    title: str = Field(..., min_length=1)
    body_html: Optional[str] = Field(default=None)
    vendor: Optional[str] = Field(default=None)
    product_type: Optional[str] = Field(default=None)
    tags: Optional[str] = Field(default=None)
    status: Optional[str] = Field(default="draft")
    variants: Optional[List[Dict[str, Any]]] = Field(default=None)
    options: Optional[List[Dict[str, Any]]] = Field(default=None)
    images: Optional[List[Dict[str, Any]]] = Field(default=None)

@mcp.tool(name="shopify_create_product")
async def shopify_create_product(params: CreateProductInput) -> str:
    """Create a new product in the Shopify store."""
    try:
        product: Dict[str, Any] = {"title": params.title}
        for field in ["body_html", "vendor", "product_type", "tags", "status", "variants", "options", "images"]:
            val = getattr(params, field)
            if val is not None:
                product[field] = val
        data = await _request("POST", "products.json", body={"product": product})
        return _fmt(data.get("product", data))
    except Exception as e:
        return _error(e)

class UpdateProductInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    product_id: int = Field(..., description="Product ID to update")
    title: Optional[str] = Field(default=None)
    body_html: Optional[str] = Field(default=None)
    vendor: Optional[str] = Field(default=None)
    product_type: Optional[str] = Field(default=None)
    tags: Optional[str] = Field(default=None)
    status: Optional[str] = Field(default=None)
    variants: Optional[List[Dict[str, Any]]] = Field(default=None)

@mcp.tool(name="shopify_update_product")
async def shopify_update_product(params: UpdateProductInput) -> str:
    """Update an existing product. Only provided fields are changed."""
    try:
        product: Dict[str, Any] = {}
        for field in ["title", "body_html", "vendor", "product_type", "tags", "status", "variants"]:
            val = getattr(params, field)
            if val is not None:
                product[field] = val
        data = await _request("PUT", f"products/{params.product_id}.json", body={"product": product})
        return _fmt(data.get("product", data))
    except Exception as e:
        return _error(e)

class DeleteProductInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_id: int = Field(..., description="Product ID to delete")

@mcp.tool(name="shopify_delete_product")
async def shopify_delete_product(params: DeleteProductInput) -> str:
    """Permanently delete a product. This cannot be undone."""
    try:
        await _request("DELETE", f"products/{params.product_id}.json")
        return f"Product {params.product_id} deleted."
    except Exception as e:
        return _error(e)

class ProductCountInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Optional[str] = Field(default=None)
    vendor: Optional[str] = Field(default=None)
    product_type: Optional[str] = Field(default=None)

@mcp.tool(name="shopify_count_products")
async def shopify_count_products(params: ProductCountInput) -> str:
    """Get the total count of products, optionally filtered."""
    try:
        p: Dict[str, Any] = {}
        for field in ["status", "vendor", "product_type"]:
            val = getattr(params, field)
            if val is not None:
                p[field] = val
        data = await _request("GET", "products/count.json", params=p)
        return _fmt(data)
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# ORDERS
# ═══════════════════════════════════════════════════════════════════════════

class ListOrdersInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    limit: Optional[int] = Field(default=50, ge=1, le=250)
    status: Optional[str] = Field(default="any")
    financial_status: Optional[str] = Field(default=None)
    fulfillment_status: Optional[str] = Field(default=None)
    since_id: Optional[int] = Field(default=None)
    created_at_min: Optional[str] = Field(default=None)
    created_at_max: Optional[str] = Field(default=None)
    fields: Optional[str] = Field(default=None)

@mcp.tool(name="shopify_list_orders")
async def shopify_list_orders(params: ListOrdersInput) -> str:
    """List orders with optional filters for status, financial/fulfillment status, and date range."""
    try:
        p: Dict[str, Any] = {"limit": params.limit, "status": params.status}
        for field in ["financial_status", "fulfillment_status", "since_id", "created_at_min", "created_at_max", "fields"]:
            val = getattr(params, field)
            if val is not None:
                p[field] = val
        data = await _request("GET", "orders.json", params=p)
        orders = data.get("orders", [])
        return _fmt({"count": len(orders), "orders": orders})
    except Exception as e:
        return _error(e)

class GetOrderInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: int = Field(..., description="The Shopify order ID")

@mcp.tool(name="shopify_get_order")
async def shopify_get_order(params: GetOrderInput) -> str:
    """Retrieve a single order by ID with full details."""
    try:
        data = await _request("GET", f"orders/{params.order_id}.json")
        return _fmt(data.get("order", data))
    except Exception as e:
        return _error(e)

class OrderCountInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Optional[str] = Field(default="any")
    financial_status: Optional[str] = Field(default=None)
    fulfillment_status: Optional[str] = Field(default=None)

@mcp.tool(name="shopify_count_orders")
async def shopify_count_orders(params: OrderCountInput) -> str:
    """Get total order count, optionally filtered."""
    try:
        p: Dict[str, Any] = {"status": params.status}
        for field in ["financial_status", "fulfillment_status"]:
            val = getattr(params, field)
            if val is not None:
                p[field] = val
        data = await _request("GET", "orders/count.json", params=p)
        return _fmt(data)
    except Exception as e:
        return _error(e)

class CloseOrderInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: int = Field(..., description="Order ID to close")

@mcp.tool(name="shopify_close_order")
async def shopify_close_order(params: CloseOrderInput) -> str:
    """Close an order (marks it as completed)."""
    try:
        data = await _request("POST", f"orders/{params.order_id}/close.json")
        return _fmt(data.get("order", data))
    except Exception as e:
        return _error(e)

class CancelOrderInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: int = Field(..., description="Order ID to cancel")
    reason: Optional[str] = Field(default=None)
    email: Optional[bool] = Field(default=True)
    restock: Optional[bool] = Field(default=False)

@mcp.tool(name="shopify_cancel_order")
async def shopify_cancel_order(params: CancelOrderInput) -> str:
    """Cancel an order. Optionally restock items and notify the customer."""
    try:
        body: Dict[str, Any] = {}
        for field in ["reason", "email", "restock"]:
            val = getattr(params, field)
            if val is not None:
                body[field] = val
        data = await _request("POST", f"orders/{params.order_id}/cancel.json", body=body)
        return _fmt(data.get("order", data))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# CUSTOMERS
# ═══════════════════════════════════════════════════════════════════════════

class ListCustomersInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    limit: Optional[int] = Field(default=50, ge=1, le=250)
    since_id: Optional[int] = Field(default=None)
    created_at_min: Optional[str] = Field(default=None)
    created_at_max: Optional[str] = Field(default=None)
    fields: Optional[str] = Field(default=None)

@mcp.tool(name="shopify_list_customers")
async def shopify_list_customers(params: ListCustomersInput) -> str:
    """List customers from the store."""
    try:
        p: Dict[str, Any] = {"limit": params.limit}
        for f in ["since_id", "created_at_min", "created_at_max", "fields"]:
            val = getattr(params, f)
            if val is not None:
                p[f] = val
        data = await _request("GET", "customers.json", params=p)
        customers = data.get("customers", [])
        return _fmt({"count": len(customers), "customers": customers})
    except Exception as e:
        return _error(e)

class SearchCustomersInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query: str = Field(..., min_length=1)
    limit: Optional[int] = Field(default=50, ge=1, le=250)

@mcp.tool(name="shopify_search_customers")
async def shopify_search_customers(params: SearchCustomersInput) -> str:
    """Search customers by name, email, or other fields."""
    try:
        data = await _request("GET", "customers/search.json", params={"query": params.query, "limit": params.limit})
        return _fmt({"count": len(data.get("customers", [])), "customers": data.get("customers", [])})
    except Exception as e:
        return _error(e)

class GetCustomerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_id: int = Field(..., description="Shopify customer ID")

@mcp.tool(name="shopify_get_customer")
async def shopify_get_customer(params: GetCustomerInput) -> str:
    """Retrieve a single customer by ID."""
    try:
        data = await _request("GET", f"customers/{params.customer_id}.json")
        return _fmt(data.get("customer", data))
    except Exception as e:
        return _error(e)

class CreateCustomerInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    first_name: Optional[str] = Field(default=None)
    last_name: Optional[str] = Field(default=None)
    email: Optional[str] = Field(default=None)
    phone: Optional[str] = Field(default=None)
    tags: Optional[str] = Field(default=None)
    note: Optional[str] = Field(default=None)
    addresses: Optional[List[Dict[str, Any]]] = Field(default=None)
    send_email_invite: Optional[bool] = Field(default=False)

@mcp.tool(name="shopify_create_customer")
async def shopify_create_customer(params: CreateCustomerInput) -> str:
    """Create a new customer."""
    try:
        customer: Dict[str, Any] = {}
        for field in ["first_name", "last_name", "email", "phone", "tags", "note", "addresses", "send_email_invite"]:
            val = getattr(params, field)
            if val is not None:
                customer[field] = val
        data = await _request("POST", "customers.json", body={"customer": customer})
        return _fmt(data.get("customer", data))
    except Exception as e:
        return _error(e)

class UpdateCustomerInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    customer_id: int = Field(..., description="Customer ID to update")
    first_name: Optional[str] = Field(default=None)
    last_name: Optional[str] = Field(default=None)
    email: Optional[str] = Field(default=None)
    phone: Optional[str] = Field(default=None)
    tags: Optional[str] = Field(default=None)
    note: Optional[str] = Field(default=None)

@mcp.tool(name="shopify_update_customer")
async def shopify_update_customer(params: UpdateCustomerInput) -> str:
    """Update an existing customer."""
    try:
        customer: Dict[str, Any] = {}
        for field in ["first_name", "last_name", "email", "phone", "tags", "note"]:
            val = getattr(params, field)
            if val is not None:
                customer[field] = val
        data = await _request("PUT", f"customers/{params.customer_id}.json", body={"customer": customer})
        return _fmt(data.get("customer", data))
    except Exception as e:
        return _error(e)

class CustomerOrdersInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_id: int = Field(..., description="Customer ID")
    limit: Optional[int] = Field(default=50, ge=1, le=250)
    status: Optional[str] = Field(default="any")

@mcp.tool(name="shopify_get_customer_orders")
async def shopify_get_customer_orders(params: CustomerOrdersInput) -> str:
    """Get all orders for a specific customer."""
    try:
        data = await _request("GET", f"customers/{params.customer_id}/orders.json", params={"limit": params.limit, "status": params.status})
        orders = data.get("orders", [])
        return _fmt({"count": len(orders), "orders": orders})
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# COLLECTIONS
# ═══════════════════════════════════════════════════════════════════════════

class ListCollectionsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: Optional[int] = Field(default=50, ge=1, le=250)
    since_id: Optional[int] = Field(default=None)
    collection_type: Optional[str] = Field(default="custom", description="'custom' or 'smart'")

@mcp.tool(name="shopify_list_collections")
async def shopify_list_collections(params: ListCollectionsInput) -> str:
    """List custom or smart collections."""
    try:
        endpoint = "custom_collections.json" if params.collection_type == "custom" else "smart_collections.json"
        p: Dict[str, Any] = {"limit": params.limit}
        if params.since_id:
            p["since_id"] = params.since_id
        data = await _request("GET", endpoint, params=p)
        key = "custom_collections" if params.collection_type == "custom" else "smart_collections"
        collections = data.get(key, [])
        return _fmt({"count": len(collections), "collections": collections})
    except Exception as e:
        return _error(e)

class GetCollectionProductsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    collection_id: int = Field(..., description="Collection ID")
    limit: Optional[int] = Field(default=50, ge=1, le=250)

@mcp.tool(name="shopify_get_collection_products")
async def shopify_get_collection_products(params: GetCollectionProductsInput) -> str:
    """Get all products in a specific collection."""
    try:
        data = await _request("GET", "products.json", params={"limit": params.limit, "collection_id": params.collection_id})
        products = data.get("products", [])
        return _fmt({"count": len(products), "products": products})
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# INVENTORY
# ═══════════════════════════════════════════════════════════════════════════

class ListInventoryLocationsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

@mcp.tool(name="shopify_list_locations")
async def shopify_list_locations(params: ListInventoryLocationsInput) -> str:
    """List all inventory locations for the store."""
    try:
        data = await _request("GET", "locations.json")
        locations = data.get("locations", [])
        return _fmt({"count": len(locations), "locations": locations})
    except Exception as e:
        return _error(e)

class GetInventoryLevelsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    location_id: Optional[int] = Field(default=None)
    inventory_item_ids: Optional[str] = Field(default=None)

@mcp.tool(name="shopify_get_inventory_levels")
async def shopify_get_inventory_levels(params: GetInventoryLevelsInput) -> str:
    """Get inventory levels for specific locations or inventory items."""
    try:
        p: Dict[str, Any] = {}
        if params.location_id:
            p["location_ids"] = params.location_id
        if params.inventory_item_ids:
            p["inventory_item_ids"] = params.inventory_item_ids
        data = await _request("GET", "inventory_levels.json", params=p)
        levels = data.get("inventory_levels", [])
        return _fmt({"count": len(levels), "inventory_levels": levels})
    except Exception as e:
        return _error(e)

class SetInventoryLevelInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inventory_item_id: int = Field(...)
    location_id: int = Field(...)
    available: int = Field(...)

@mcp.tool(name="shopify_set_inventory_level")
async def shopify_set_inventory_level(params: SetInventoryLevelInput) -> str:
    """Set the available inventory for an item at a location."""
    try:
        body = {"inventory_item_id": params.inventory_item_id, "location_id": params.location_id, "available": params.available}
        data = await _request("POST", "inventory_levels/set.json", body=body)
        return _fmt(data.get("inventory_level", data))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# FULFILLMENTS
# ═══════════════════════════════════════════════════════════════════════════

class ListFulfillmentsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: int = Field(..., description="Order ID")
    limit: Optional[int] = Field(default=50, ge=1, le=250)

@mcp.tool(name="shopify_list_fulfillments")
async def shopify_list_fulfillments(params: ListFulfillmentsInput) -> str:
    """List fulfillments for a specific order."""
    try:
        data = await _request("GET", f"orders/{params.order_id}/fulfillments.json", params={"limit": params.limit})
        fulfillments = data.get("fulfillments", [])
        return _fmt({"count": len(fulfillments), "fulfillments": fulfillments})
    except Exception as e:
        return _error(e)

class CreateFulfillmentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: int = Field(...)
    location_id: int = Field(...)
    tracking_number: Optional[str] = Field(default=None)
    tracking_company: Optional[str] = Field(default=None)
    tracking_url: Optional[str] = Field(default=None)
    line_items: Optional[List[Dict[str, Any]]] = Field(default=None)
    notify_customer: Optional[bool] = Field(default=True)

@mcp.tool(name="shopify_create_fulfillment")
async def shopify_create_fulfillment(params: CreateFulfillmentInput) -> str:
    """Create a fulfillment for an order (ship items)."""
    try:
        fulfillment: Dict[str, Any] = {"location_id": params.location_id}
        for field in ["tracking_number", "tracking_company", "tracking_url", "line_items", "notify_customer"]:
            val = getattr(params, field)
            if val is not None:
                fulfillment[field] = val
        data = await _request("POST", f"orders/{params.order_id}/fulfillments.json", body={"fulfillment": fulfillment})
        return _fmt(data.get("fulfillment", data))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# SHOP INFO
# ═══════════════════════════════════════════════════════════════════════════

class EmptyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

@mcp.tool(name="shopify_get_shop")
async def shopify_get_shop(params: EmptyInput) -> str:
    """Get store information: name, domain, plan, currency, timezone, etc."""
    try:
        data = await _request("GET", "shop.json")
        return _fmt(data.get("shop", data))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# WEBHOOKS
# ═══════════════════════════════════════════════════════════════════════════

class ListWebhooksInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: Optional[int] = Field(default=50, ge=1, le=250)
    topic: Optional[str] = Field(default=None)

@mcp.tool(name="shopify_list_webhooks")
async def shopify_list_webhooks(params: ListWebhooksInput) -> str:
    """List configured webhooks."""
    try:
        p: Dict[str, Any] = {"limit": params.limit}
        if params.topic:
            p["topic"] = params.topic
        data = await _request("GET", "webhooks.json", params=p)
        webhooks = data.get("webhooks", [])
        return _fmt({"count": len(webhooks), "webhooks": webhooks})
    except Exception as e:
        return _error(e)

class CreateWebhookInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    topic: str = Field(...)
    address: str = Field(...)
    format: Optional[str] = Field(default="json")

@mcp.tool(name="shopify_create_webhook")
async def shopify_create_webhook(params: CreateWebhookInput) -> str:
    """Create a new webhook subscription."""
    try:
        data = await _request("POST", "webhooks.json", body={"webhook": {"topic": params.topic, "address": params.address, "format": params.format}})
        return _fmt(data.get("webhook", data))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# PAGES  ← NEW
# ═══════════════════════════════════════════════════════════════════════════

class ListPagesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: Optional[int] = Field(default=50, ge=1, le=250)
    since_id: Optional[int] = Field(default=None)
    title: Optional[str] = Field(default=None, description="Filter by page title")
    published_status: Optional[str] = Field(default=None, description="published, unpublished, any")
    fields: Optional[str] = Field(default=None, description="Comma-separated fields to return")

@mcp.tool(name="shopify_list_pages")
async def shopify_list_pages(params: ListPagesInput) -> str:
    """List all store pages (About Us, Contact, FAQ, etc.)."""
    try:
        p: Dict[str, Any] = {"limit": params.limit}
        for field in ["since_id", "title", "published_status", "fields"]:
            val = getattr(params, field)
            if val is not None:
                p[field] = val
        data = await _request("GET", "pages.json", params=p)
        pages = data.get("pages", [])
        return _fmt({"count": len(pages), "pages": pages})
    except Exception as e:
        return _error(e)

class GetPageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    page_id: int = Field(..., description="The Shopify page ID")

@mcp.tool(name="shopify_get_page")
async def shopify_get_page(params: GetPageInput) -> str:
    """Get a single page by ID including full HTML body content."""
    try:
        data = await _request("GET", f"pages/{params.page_id}.json")
        return _fmt(data.get("page", data))
    except Exception as e:
        return _error(e)

class CountPagesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    published_status: Optional[str] = Field(default=None, description="published, unpublished, any")

@mcp.tool(name="shopify_count_pages")
async def shopify_count_pages(params: CountPagesInput) -> str:
    """Count total store pages."""
    try:
        p: Dict[str, Any] = {}
        if params.published_status:
            p["published_status"] = params.published_status
        data = await _request("GET", "pages/count.json", params=p)
        return _fmt(data)
    except Exception as e:
        return _error(e)

class CreatePageInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    title: str = Field(..., min_length=1)
    body_html: Optional[str] = Field(default=None, description="HTML content of the page")
    handle: Optional[str] = Field(default=None, description="URL slug, e.g. 'about-us'")
    published: Optional[bool] = Field(default=True)
    metafield: Optional[Dict[str, Any]] = Field(default=None)

@mcp.tool(name="shopify_create_page")
async def shopify_create_page(params: CreatePageInput) -> str:
    """Create a new store page."""
    try:
        page: Dict[str, Any] = {"title": params.title}
        for field in ["body_html", "handle", "published", "metafield"]:
            val = getattr(params, field)
            if val is not None:
                page[field] = val
        data = await _request("POST", "pages.json", body={"page": page})
        return _fmt(data.get("page", data))
    except Exception as e:
        return _error(e)

class UpdatePageInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    page_id: int = Field(..., description="Page ID to update")
    title: Optional[str] = Field(default=None)
    body_html: Optional[str] = Field(default=None)
    handle: Optional[str] = Field(default=None)
    published: Optional[bool] = Field(default=None)

@mcp.tool(name="shopify_update_page")
async def shopify_update_page(params: UpdatePageInput) -> str:
    """Update an existing store page."""
    try:
        page: Dict[str, Any] = {}
        for field in ["title", "body_html", "handle", "published"]:
            val = getattr(params, field)
            if val is not None:
                page[field] = val
        data = await _request("PUT", f"pages/{params.page_id}.json", body={"page": page})
        return _fmt(data.get("page", data))
    except Exception as e:
        return _error(e)

class DeletePageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    page_id: int = Field(..., description="Page ID to delete")

@mcp.tool(name="shopify_delete_page")
async def shopify_delete_page(params: DeletePageInput) -> str:
    """Delete a store page permanently."""
    try:
        await _request("DELETE", f"pages/{params.page_id}.json")
        return f"Page {params.page_id} deleted."
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# THEMES  ← NEW
# ═══════════════════════════════════════════════════════════════════════════

class ListThemesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

@mcp.tool(name="shopify_list_themes")
async def shopify_list_themes(params: ListThemesInput) -> str:
    """List all themes in the store. The active theme has role='main'."""
    try:
        data = await _request("GET", "themes.json")
        themes = data.get("themes", [])
        return _fmt({"count": len(themes), "themes": themes})
    except Exception as e:
        return _error(e)

class GetThemeAssetsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    theme_id: int = Field(..., description="The theme ID")

@mcp.tool(name="shopify_list_theme_assets")
async def shopify_list_theme_assets(params: GetThemeAssetsInput) -> str:
    """List all files/assets in a theme (Liquid, CSS, JS, JSON, images)."""
    try:
        data = await _request("GET", f"themes/{params.theme_id}/assets.json")
        assets = data.get("assets", [])
        return _fmt({"count": len(assets), "assets": assets})
    except Exception as e:
        return _error(e)

class GetThemeAssetInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    theme_id: int = Field(..., description="The theme ID")
    asset_key: str = Field(..., description="Asset key, e.g. 'templates/index.json' or 'sections/header.liquid'")

@mcp.tool(name="shopify_get_theme_asset")
async def shopify_get_theme_asset(params: GetThemeAssetInput) -> str:
    """Read the full content of a theme file (Liquid, JSON sections, CSS, JS)."""
    try:
        data = await _request("GET", f"themes/{params.theme_id}/assets.json", params={"asset[key]": params.asset_key})
        return _fmt(data.get("asset", data))
    except Exception as e:
        return _error(e)

class UpdateThemeAssetInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    theme_id: int = Field(..., description="The theme ID")
    asset_key: str = Field(..., description="Asset key, e.g. 'templates/index.json'")
    value: Optional[str] = Field(default=None, description="Text content of the file (for Liquid/JSON/CSS/JS)")
    src: Optional[str] = Field(default=None, description="URL to download asset from (for images)")
    attachment: Optional[str] = Field(default=None, description="Base64-encoded content (for binary files)")

@mcp.tool(name="shopify_update_theme_asset")
async def shopify_update_theme_asset(params: UpdateThemeAssetInput) -> str:
    """Create or update a theme file. Use value for text files (Liquid, JSON, CSS), attachment for binary."""
    try:
        asset: Dict[str, Any] = {"key": params.asset_key}
        if params.value is not None:
            asset["value"] = params.value
        if params.src is not None:
            asset["src"] = params.src
        if params.attachment is not None:
            asset["attachment"] = params.attachment
        data = await _request("PUT", f"themes/{params.theme_id}/assets.json", body={"asset": asset})
        return _fmt(data.get("asset", data))
    except Exception as e:
        return _error(e)

class DeleteThemeAssetInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    theme_id: int = Field(..., description="The theme ID")
    asset_key: str = Field(..., description="Asset key to delete")

@mcp.tool(name="shopify_delete_theme_asset")
async def shopify_delete_theme_asset(params: DeleteThemeAssetInput) -> str:
    """Delete a theme asset/file."""
    try:
        await _request("DELETE", f"themes/{params.theme_id}/assets.json?asset[key]={params.asset_key}")
        return f"Asset '{params.asset_key}' deleted from theme {params.theme_id}."
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# FILES (Shopify CDN)  ← NEW
# ═══════════════════════════════════════════════════════════════════════════

class ListFilesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: Optional[int] = Field(default=50, ge=1, le=250)
    fields: Optional[str] = Field(default=None)

@mcp.tool(name="shopify_list_files")
async def shopify_list_files(params: ListFilesInput) -> str:
    """List files uploaded to Shopify CDN (store files, images)."""
    try:
        p: Dict[str, Any] = {"limit": params.limit}
        if params.fields:
            p["fields"] = params.fields
        data = await _request("GET", "files.json", params=p)
        files = data.get("files", [])
        return _fmt({"count": len(files), "files": files})
    except Exception as e:
        return _error(e)

class UploadFileInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    src: str = Field(..., description="Public URL of the file to upload to Shopify CDN")
    filename: Optional[str] = Field(default=None, description="Optional filename override")
    content_type: Optional[str] = Field(default=None, description="MIME type, e.g. image/jpeg")

@mcp.tool(name="shopify_upload_file")
async def shopify_upload_file(params: UploadFileInput) -> str:
    """Upload a file to Shopify CDN from a public URL. Returns CDN URL."""
    try:
        file_obj: Dict[str, Any] = {"src": params.src}
        if params.filename:
            file_obj["filename"] = params.filename
        if params.content_type:
            file_obj["content_type"] = params.content_type
        data = await _request("POST", "files.json", body={"file": file_obj})
        return _fmt(data.get("file", data))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# BLOGS & ARTICLES  ← NEW
# ═══════════════════════════════════════════════════════════════════════════

class ListBlogsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: Optional[int] = Field(default=50, ge=1, le=250)

@mcp.tool(name="shopify_list_blogs")
async def shopify_list_blogs(params: ListBlogsInput) -> str:
    """List all blogs in the store."""
    try:
        data = await _request("GET", "blogs.json", params={"limit": params.limit})
        blogs = data.get("blogs", [])
        return _fmt({"count": len(blogs), "blogs": blogs})
    except Exception as e:
        return _error(e)

class ListArticlesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    blog_id: int = Field(..., description="Blog ID to list articles from")
    limit: Optional[int] = Field(default=50, ge=1, le=250)
    published_status: Optional[str] = Field(default=None, description="published, unpublished, any")

@mcp.tool(name="shopify_list_articles")
async def shopify_list_articles(params: ListArticlesInput) -> str:
    """List articles in a blog."""
    try:
        p: Dict[str, Any] = {"limit": params.limit}
        if params.published_status:
            p["published_status"] = params.published_status
        data = await _request("GET", f"blogs/{params.blog_id}/articles.json", params=p)
        articles = data.get("articles", [])
        return _fmt({"count": len(articles), "articles": articles})
    except Exception as e:
        return _error(e)

class CreateArticleInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    blog_id: int = Field(...)
    title: str = Field(..., min_length=1)
    body_html: Optional[str] = Field(default=None)
    author: Optional[str] = Field(default=None)
    tags: Optional[str] = Field(default=None)
    published: Optional[bool] = Field(default=True)
    image: Optional[Dict[str, Any]] = Field(default=None, description="Image object with src URL")

@mcp.tool(name="shopify_create_article")
async def shopify_create_article(params: CreateArticleInput) -> str:
    """Create a new blog article."""
    try:
        article: Dict[str, Any] = {"title": params.title}
        for field in ["body_html", "author", "tags", "published", "image"]:
            val = getattr(params, field)
            if val is not None:
                article[field] = val
        data = await _request("POST", f"blogs/{params.blog_id}/articles.json", body={"article": article})
        return _fmt(data.get("article", data))
    except Exception as e:
        return _error(e)

class UpdateArticleInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    blog_id: int = Field(...)
    article_id: int = Field(...)
    title: Optional[str] = Field(default=None)
    body_html: Optional[str] = Field(default=None)
    author: Optional[str] = Field(default=None)
    tags: Optional[str] = Field(default=None)
    published: Optional[bool] = Field(default=None)

@mcp.tool(name="shopify_update_article")
async def shopify_update_article(params: UpdateArticleInput) -> str:
    """Update an existing blog article."""
    try:
        article: Dict[str, Any] = {}
        for field in ["title", "body_html", "author", "tags", "published"]:
            val = getattr(params, field)
            if val is not None:
                article[field] = val
        data = await _request("PUT", f"blogs/{params.blog_id}/articles/{params.article_id}.json", body={"article": article})
        return _fmt(data.get("article", data))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# METAFIELDS  ← NEW
# ═══════════════════════════════════════════════════════════════════════════

class ListMetafieldsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resource: str = Field(..., description="Resource type: products, collections, customers, orders, pages, shop")
    resource_id: Optional[int] = Field(default=None, description="Resource ID (omit for shop-level metafields)")
    namespace: Optional[str] = Field(default=None, description="Filter by namespace")
    limit: Optional[int] = Field(default=50, ge=1, le=250)

@mcp.tool(name="shopify_list_metafields")
async def shopify_list_metafields(params: ListMetafieldsInput) -> str:
    """List metafields for any resource (product, collection, page, order, shop, etc.)."""
    try:
        if params.resource_id:
            endpoint = f"{params.resource}/{params.resource_id}/metafields.json"
        else:
            endpoint = "metafields.json"
        p: Dict[str, Any] = {"limit": params.limit}
        if params.namespace:
            p["namespace"] = params.namespace
        data = await _request("GET", endpoint, params=p)
        metafields = data.get("metafields", [])
        return _fmt({"count": len(metafields), "metafields": metafields})
    except Exception as e:
        return _error(e)

class SetMetafieldInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    resource: str = Field(..., description="Resource type: products, collections, customers, orders, pages")
    resource_id: int = Field(..., description="Resource ID")
    namespace: str = Field(..., description="Metafield namespace, e.g. 'custom'")
    key: str = Field(..., description="Metafield key")
    value: str = Field(..., description="Metafield value")
    type: str = Field(..., description="Value type: single_line_text_field, multi_line_text_field, integer, json, etc.")

@mcp.tool(name="shopify_set_metafield")
async def shopify_set_metafield(params: SetMetafieldInput) -> str:
    """Create or update a metafield on any resource."""
    try:
        body = {
            "metafield": {
                "namespace": params.namespace,
                "key": params.key,
                "value": params.value,
                "type": params.type,
            }
        }
        data = await _request("POST", f"{params.resource}/{params.resource_id}/metafields.json", body=body)
        return _fmt(data.get("metafield", data))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# REDIRECTS  ← NEW
# ═══════════════════════════════════════════════════════════════════════════

class ListRedirectsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: Optional[int] = Field(default=50, ge=1, le=250)
    path: Optional[str] = Field(default=None, description="Filter by source path")

@mcp.tool(name="shopify_list_redirects")
async def shopify_list_redirects(params: ListRedirectsInput) -> str:
    """List URL redirects configured in the store."""
    try:
        p: Dict[str, Any] = {"limit": params.limit}
        if params.path:
            p["path"] = params.path
        data = await _request("GET", "redirects.json", params=p)
        redirects = data.get("redirects", [])
        return _fmt({"count": len(redirects), "redirects": redirects})
    except Exception as e:
        return _error(e)

class CreateRedirectInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    path: str = Field(..., description="Source path, e.g. /old-page")
    target: str = Field(..., description="Target URL or path, e.g. /new-page")

@mcp.tool(name="shopify_create_redirect")
async def shopify_create_redirect(params: CreateRedirectInput) -> str:
    """Create a URL redirect."""
    try:
        data = await _request("POST", "redirects.json", body={"redirect": {"path": params.path, "target": params.target}})
        return _fmt(data.get("redirect", data))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# SCRIPT TAGS  ← NEW
# ═══════════════════════════════════════════════════════════════════════════

class ListScriptTagsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: Optional[int] = Field(default=50, ge=1, le=250)

@mcp.tool(name="shopify_list_script_tags")
async def shopify_list_script_tags(params: ListScriptTagsInput) -> str:
    """List all script tags injected into the storefront."""
    try:
        data = await _request("GET", "script_tags.json", params={"limit": params.limit})
        script_tags = data.get("script_tags", [])
        return _fmt({"count": len(script_tags), "script_tags": script_tags})
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# PRODUCT IMAGES  ← NEW
# ═══════════════════════════════════════════════════════════════════════════

class ListProductImagesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_id: int = Field(..., description="Product ID")

@mcp.tool(name="shopify_list_product_images")
async def shopify_list_product_images(params: ListProductImagesInput) -> str:
    """List all images for a specific product."""
    try:
        data = await _request("GET", f"products/{params.product_id}/images.json")
        images = data.get("images", [])
        return _fmt({"count": len(images), "images": images})
    except Exception as e:
        return _error(e)

class AddProductImageInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    product_id: int = Field(..., description="Product ID")
    src: str = Field(..., description="Public URL of the image to add")
    alt: Optional[str] = Field(default=None, description="Alt text for the image")
    position: Optional[int] = Field(default=None, description="Image position (1 = first/main)")
    variant_ids: Optional[List[int]] = Field(default=None, description="Variant IDs to attach this image to")

@mcp.tool(name="shopify_add_product_image")
async def shopify_add_product_image(params: AddProductImageInput) -> str:
    """Add an image to a product from a public URL."""
    try:
        image: Dict[str, Any] = {"src": params.src}
        for field in ["alt", "position", "variant_ids"]:
            val = getattr(params, field)
            if val is not None:
                image[field] = val
        data = await _request("POST", f"products/{params.product_id}/images.json", body={"image": image})
        return _fmt(data.get("image", data))
    except Exception as e:
        return _error(e)

class DeleteProductImageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_id: int = Field(..., description="Product ID")
    image_id: int = Field(..., description="Image ID to delete")

@mcp.tool(name="shopify_delete_product_image")
async def shopify_delete_product_image(params: DeleteProductImageInput) -> str:
    """Delete an image from a product."""
    try:
        await _request("DELETE", f"products/{params.product_id}/images/{params.image_id}.json")
        return f"Image {params.image_id} deleted from product {params.product_id}."
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# COLLECTIONS MANAGEMENT  ← NEW
# ═══════════════════════════════════════════════════════════════════════════

class CreateCollectionInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    title: str = Field(..., min_length=1)
    body_html: Optional[str] = Field(default=None)
    handle: Optional[str] = Field(default=None)
    published: Optional[bool] = Field(default=True)
    image: Optional[Dict[str, Any]] = Field(default=None, description="Image object with src URL")
    sort_order: Optional[str] = Field(default=None, description="manual, best-selling, alpha-asc, etc.")

@mcp.tool(name="shopify_create_collection")
async def shopify_create_collection(params: CreateCollectionInput) -> str:
    """Create a new custom collection."""
    try:
        collection: Dict[str, Any] = {"title": params.title}
        for field in ["body_html", "handle", "published", "image", "sort_order"]:
            val = getattr(params, field)
            if val is not None:
                collection[field] = val
        data = await _request("POST", "custom_collections.json", body={"custom_collection": collection})
        return _fmt(data.get("custom_collection", data))
    except Exception as e:
        return _error(e)

class UpdateCollectionInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    collection_id: int = Field(...)
    title: Optional[str] = Field(default=None)
    body_html: Optional[str] = Field(default=None)
    handle: Optional[str] = Field(default=None)
    published: Optional[bool] = Field(default=None)
    image: Optional[Dict[str, Any]] = Field(default=None)
    sort_order: Optional[str] = Field(default=None)

@mcp.tool(name="shopify_update_collection")
async def shopify_update_collection(params: UpdateCollectionInput) -> str:
    """Update an existing custom collection."""
    try:
        collection: Dict[str, Any] = {}
        for field in ["title", "body_html", "handle", "published", "image", "sort_order"]:
            val = getattr(params, field)
            if val is not None:
                collection[field] = val
        data = await _request("PUT", f"custom_collections/{params.collection_id}.json", body={"custom_collection": collection})
        return _fmt(data.get("custom_collection", data))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# PRICE RULES & DISCOUNTS  ← NEW
# ═══════════════════════════════════════════════════════════════════════════

class ListPriceRulesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: Optional[int] = Field(default=50, ge=1, le=250)

@mcp.tool(name="shopify_list_price_rules")
async def shopify_list_price_rules(params: ListPriceRulesInput) -> str:
    """List all price rules (discount campaigns)."""
    try:
        data = await _request("GET", "price_rules.json", params={"limit": params.limit})
        rules = data.get("price_rules", [])
        return _fmt({"count": len(rules), "price_rules": rules})
    except Exception as e:
        return _error(e)

class ListDiscountCodesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    price_rule_id: int = Field(..., description="Price rule ID to list discount codes for")
    limit: Optional[int] = Field(default=50, ge=1, le=250)

@mcp.tool(name="shopify_list_discount_codes")
async def shopify_list_discount_codes(params: ListDiscountCodesInput) -> str:
    """List discount codes for a price rule."""
    try:
        data = await _request("GET", f"price_rules/{params.price_rule_id}/discount_codes.json", params={"limit": params.limit})
        codes = data.get("discount_codes", [])
        return _fmt({"count": len(codes), "discount_codes": codes})
    except Exception as e:
        return _error(e)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport=MCP_TRANSPORT)
