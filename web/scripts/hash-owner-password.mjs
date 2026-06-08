import { hash } from "bcryptjs";

const password = process.env.OWNER_PASSWORD;

if (!password) {
  console.error("Set OWNER_PASSWORD before running this script.");
  process.exit(1);
}

console.log(await hash(password, 12));
